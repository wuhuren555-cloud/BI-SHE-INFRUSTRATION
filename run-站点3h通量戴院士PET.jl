using Pkg
Pkg.activate(raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\SoilDiffEqs.jl-master")
Pkg.instantiate()

using Base.Threads
using SoilDifferentialEquations, Ipaper
using DataFrames, CSV, Dates, XLSX
using Plots, Statistics, Random, StatsBase
using NCDatasets 

# ====================================================================
# 1. 路径配置与目录创建
# ====================================================================
ENV["GKSwstype"] = "nul"
Random.seed!(42)
default(fontfamily="sans-serif") 

path_xgb_preds = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\XGBoost_13Features_调参\All_Sites_Daily_Infiltration_Predictions_Continuous.csv"
file_sm        = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\sm\CHinaSM_hourly_2018-2019.nc"
path_era5_et   = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\ERA5_Hourly_PET_ET_2018_2019.csv"

# 🌟 戴院士参数文件
path_dai2019   = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\Dai2019_SoilProperties_sp2702.xlsx"

config_file    = "my_config正向计算.yaml"
output_dir     = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\Physics_Forward_戴PET-改2/"
if !isdir(output_dir) mkpath(output_dir) end
plot_dir = joinpath(output_dir, "All_Valid_Sites_SWS_Plots")
if !isdir(plot_dir) mkpath(plot_dir) end

parse_dt(x::DateTime) = x
parse_dt(x::AbstractString) = tryparse(DateTime, x) === nothing ? DateTime(x, "yyyy-mm-dd HH:MM:SS") : DateTime(x)
parse_dt(::Missing) = missing

# ====================================================================
# 2. 核心物理计算函数 (BEPS胁迫 + 指数根系) [保持不变]
# ====================================================================

function generate_root_fractions(dz_layers::Vector{Float64}, beta::Float64)
    N = length(dz_layers)
    raw_fracs = zeros(N)
    d_top = 0.0
    for i in 1:N
        d_bot = d_top + dz_layers[i]
        raw_fracs[i] = (beta ^ d_top) - (beta ^ d_bot)
        d_top = d_bot
    end
    return raw_fracs ./ sum(raw_fracs)
end

function compute_BEPS_sink!(sink, θ, param, root_fractions, Tsoil_p, ET_pot_cm_h, dz_layers, N_layers)
    ψ_min_tension = 3300.0  
    alpha = 1.5             
    t1 = -0.02
    t2 = 2.0

    w_root = zeros(N_layers)
    w_norm = zeros(N_layers)
    f_stress = zeros(N_layers)
    f_temp = zeros(N_layers)

    for i in 1:N_layers
        θ_ratio = max(0.05, min(1.0, θ[i] / param.θ_sat[i])) 
        psi_base = hasproperty(param, :ψ_e) ? param.ψ_e[i] : param.ψ_sat[i]
        
        ψ_tension = min(1000000.0, abs(psi_base * (θ_ratio ^ (-param.b[i]))))
        
        f_stress[i] = ψ_tension > ψ_min_tension ? 1.0 / (1.0 + ((ψ_tension - ψ_min_tension) / ψ_min_tension)^alpha) : 1.0
        f_temp[i] = Tsoil_p[i] > 0.0 ? 1.0 - exp(t1 * Tsoil_p[i]^t2) : 0.0

        f_stress[i] *= f_temp[i]
        w_root[i] = root_fractions[i] * f_stress[i]
    end

    w_root_sum = sum(w_root)
    f_soilwater = 0.0

    if w_root_sum < 1e-6
        f_soilwater = 0.1
        w_norm .= 0.0
    else
        f_stress_sum = 0.0
        for i in 1:N_layers
            w_norm[i] = w_root[i] / w_root_sum
            f_stress_sum += f_stress[i] * w_norm[i]
        end
        f_soilwater = max(0.1, f_stress_sum)
    end

    for i in 1:N_layers
        sink[i] = (ET_pot_cm_h * f_soilwater * w_norm[i]) / dz_layers[i]
    end
end

# ====================================================================
# 3. 数据读取与预处理 (精准匹配 DataFrame)
# ====================================================================
println("📌 [1/4] 正在加载多源驱动数据与参数...")

df_preds  = CSV.read(path_xgb_preds, DataFrame)
df_preds.date_only = Date.(df_preds.date_only)

df_et = CSV.read(path_era5_et, DataFrame)

# 🌟 精准锁定
et_time_col = :datetime
et_val_col  = :PET_mm
println("🔍 已成功锁定时间列 [$et_time_col] 和 大气潜在蒸发列 [$et_val_col]")

df_et.date_only = Date.(parse_dt.(df_et[!, et_time_col]))
df_et_daily = combine(groupby(df_et, [:site, :date_only]), et_val_col => (x -> sum(skipmissing(x))) => :Daily_ET_mm)

# 🌟 加载戴院士参数
xf = XLSX.readxlsx(path_dai2019)
df_Ksat = DataFrame(XLSX.gettable(xf["Ksat"])); df_Ksat.site = string.(df_Ksat.site)
df_θsat = DataFrame(XLSX.gettable(xf["θsat"])); df_θsat.site = string.(df_θsat.site)
df_λ    = DataFrame(XLSX.gettable(xf["λ"]));    df_λ.site    = string.(df_λ.site)
df_ψsat = DataFrame(XLSX.gettable(xf["ψsat"])); df_ψsat.site = string.(df_ψsat.site)

excel_depths = [2.5, 10.0, 22.5, 45.0, 80.0, 150.0] 
excel_cols   = ["l1", "l2", "l3", "l4", "l5", "l6"]
depth_labels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
dz_layers    = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0]

ds = NCDataset(file_sm, "r")
all_sites_raw = ds["site"][:]  
dates_nc = ds["time"][:]       
dates_hourly = DateTime.(dates_nc) 
depth_cols = ["depth_10", "depth_20", "depth_30", "depth_40", "depth_50", "depth_60", "depth_80", "depth_100"]

data_dict = Dict{String, Matrix{Union{Missing, Float64}}}()
sm_var = ds["SM"]
dim_names = dimnames(sm_var)
t_idx, d_idx, s_idx = findfirst(isequal("time"), dim_names), findfirst(isequal("depth"), dim_names), findfirst(isequal("site"), dim_names)
raw_sm_perm = permutedims(sm_var[:, :, :], (t_idx, s_idx, d_idx))
for (j, col) in enumerate(depth_cols) data_dict[col] = raw_sm_perm[:, :, j] end
close(ds)

valid_sites = intersect(unique(df_preds.site), all_sites_raw, unique(df_et_daily.site), df_Ksat.site)
total_sites = length(valid_sites)
plot_sites_list = valid_sites 

site_metrics_records = Dict{String, Vector{Float64}}()
all_sim_dfs = DataFrame[]  # 增加时间序列存储，以便导出CSV
const io_lock = ReentrantLock()

# ====================================================================
# 4. 初始化物理引擎
# ====================================================================
println("⚙️ [2/4] 正在初始化理查兹偏微分方程引擎...")
forward_config = load_config(config_file)
forward_config.itop = 1           
forward_config.z_bound_top = 1    
forward_config.col_obs_start = 4  
forward_config.method_retention = "Campbell"

# ====================================================================
# 5. 核心循环：多线程连续物理演化
# ====================================================================
println("🚀 [3/4] 启动 PET驱动BEPS多线程物理演化 (共 $total_sites 个站点) ...")
println("⚡ 当前 Julia 激活的线程数: $(Threads.nthreads())")

root_fractions = generate_root_fractions(dz_layers, 0.96)
println("🌱 使用 Jackson(1996) 生成的根系分布: ", round.(root_fractions, digits=3))

Threads.@threads for idx in 1:total_sites
    SITE_NAME = valid_sites[idx]
    
    df_site_preds = filter(row -> row.site == SITE_NAME, df_preds)
    df_site_et    = filter(row -> row.site == SITE_NAME, df_et_daily)
    
    local current_s_idx = findfirst(isequal(SITE_NAME), all_sites_raw)
    df_obs_hourly = DataFrame(time = dates_hourly)
    for col in depth_cols df_obs_hourly[!, col] = data_dict[col][:, current_s_idx] end
    
    df_obs_eod = filter(row -> hour(row.time) == 23, df_obs_hourly)
    df_obs_eod.date_only = Date.(df_obs_eod.time)
    for col in depth_cols df_obs_eod[!, col] .= df_obs_eod[!, col] ./ 100.0 end
    
    # ==================================================================
    # 🌟 核心补全：构建绝对连续的时间轴，防止间歇性断流！
    # ==================================================================
    start_date = min(minimum(df_site_preds.date_only), minimum(df_site_et.date_only))
    end_date   = max(maximum(df_site_preds.date_only), maximum(df_site_et.date_only))
    continuous_dates = collect(start_date:Day(1):end_date)
    
    df_merged = DataFrame(site = SITE_NAME, date_only = continuous_dates)
    df_merged = leftjoin(df_merged, df_site_preds, on=[:site, :date_only])
    df_merged = leftjoin(df_merged, df_obs_eod, on=:date_only)
    df_merged = leftjoin(df_merged, df_site_et, on=[:site, :date_only])
    sort!(df_merged, :date_only)
    
    if nrow(df_merged) < 10 continue end 
    
    dummy_obs = zeros(Float64, nrow(df_merged), 8) 
    
    local_config = deepcopy(forward_config)
    soil, _, _, _ = SoilDifferentialEquations.setup(local_config, dummy_obs)
    if hasproperty(soil, :ibeg) soil.ibeg = 1 end
    
    row_Ksat = filter(row -> row.site == SITE_NAME, df_Ksat)
    row_θsat = filter(row -> row.site == SITE_NAME, df_θsat)
    row_λ    = filter(row -> row.site == SITE_NAME, df_λ)
    row_ψsat = filter(row -> row.site == SITE_NAME, df_ψsat)
    
    # 🌟 修复：加入 ismissing 保护，防止戴院士 Excel 空白值报错
    for i in 1:soil.N
        idx_closest = argmin(abs.(excel_depths .- depth_labels[i]))
        col_name = excel_cols[idx_closest]
        
        val_Ksat = row_Ksat[1, col_name]
        soil.param.Ksat[i] = ismissing(val_Ksat) ? (i > 1 ? soil.param.Ksat[i-1] : (10.0 / 24.0)) : Float64(val_Ksat) / 24.0 
        
        val_theta = row_θsat[1, col_name]
        if ismissing(val_theta)
            soil.param.θ_sat[i] = i > 1 ? soil.param.θ_sat[i-1] : 0.45
        else
            vt = Float64(val_theta)
            soil.param.θ_sat[i] = vt > 1.0 ? vt / 100.0 : vt
        end
        
        val_λ_raw = row_λ[1, col_name]
        if ismissing(val_λ_raw)
            soil.param.b[i] = i > 1 ? soil.param.b[i-1] : 5.0
        else
            vl = Float64(val_λ_raw)
            soil.param.b[i] = vl > 1.0 ? vl : (1.0 / vl) 
        end
        
        val_psi = row_ψsat[1, col_name]
        if ismissing(val_psi)
            safe_psi_cm = i > 1 ? (hasproperty(soil.param, :ψ_e) ? soil.param.ψ_e[i-1] : soil.param.ψ_sat[i-1]) : -15.0
        else
            raw_psi = Float64(val_psi)
            safe_psi_cm = abs(raw_psi) > 100.0 ? (-abs(raw_psi) / 10.0) : -abs(raw_psi)
        end
        
        if hasproperty(soil.param, :ψ_e)
            soil.param.ψ_e[i] = safe_psi_cm
        else
            soil.param.ψ_sat[i] = safe_psi_cm
        end
    end
    try SoilDifferentialEquations.Update_SoilParam_Param!(soil.param) catch; end

    obs_matrix = Matrix(df_merged[:, depth_cols])
    for i in 1:soil.N
        raw_init = ismissing(obs_matrix[1, i]) ? (soil.param.θ_sat[i] * 0.5) : obs_matrix[1, i]
        safe_init = max(soil.param.θ_sat[i] * 0.10, min(soil.param.θ_sat[i] - 1e-4, raw_init))
        soil.θ[i] = soil.θ_prev[i] = safe_init
    end
    try SoilDifferentialEquations.cal_ψ!(soil, soil.θ) catch; end
    
    sim_profile = zeros(Float64, nrow(df_merged), soil.N)
    for i in 1:soil.N sim_profile[1, i] = Float64(soil.θ[i]) end
    sink = zeros(Float64, soil.N) 
    is_diverged = false
    
    dt_hours = soil.dt / 3600.0 
    steps_per_day = max(1, round(Int, 24.0 / dt_hours))
    Tsoil_p_mock = fill(15.0, soil.N)
    
    for t in 2:nrow(df_merged)
        daily_inf = coalesce(df_merged.Predicted_Infiltration[t], 0.0)
        Q0_cm_h = -1.0 * (daily_inf / 10.0) / 24.0  
        
        # 🌟 获取大气驱动的绝对上限: PET (即使不下雨，也会正常读取耗水)
        daily_pet = coalesce(df_merged.Daily_ET_mm[t], 0.0)
        ET_pot_cm_h = (daily_pet / 10.0) / 24.0 
        
        try
            for _ in 1:steps_per_day
                compute_BEPS_sink!(sink, soil.θ, soil.param, root_fractions, Tsoil_p_mock, ET_pot_cm_h, dz_layers, soil.N)
                
                SoilDifferentialEquations.soil_moisture_Q0!(soil, sink, Q0_cm_h)
                
                for i in 1:soil.N
                    if soil.θ[i] > soil.param.θ_sat[i] - 1e-4
                        soil.θ[i] = soil.param.θ_sat[i] - 1e-4
                    elseif soil.θ[i] < soil.param.θ_sat[i] * 0.05 
                        soil.θ[i] = soil.param.θ_sat[i] * 0.05
                    end
                end
                
                SoilDifferentialEquations.cal_ψ!(soil, soil.θ)
                for i in 1:soil.N soil.θ_prev[i] = soil.θ[i] end
            end
        catch e
            lock(io_lock) do
                println("    ⚠️ 站点 $SITE_NAME 报错: ", e)
            end
            for i in 1:soil.N soil.θ[i] = soil.θ_prev[i] end
            try SoilDifferentialEquations.cal_ψ!(soil, soil.θ) catch; end
        end
        
        if any(isnan.(soil.θ)) || any(isinf.(soil.θ))
            is_diverged = true
            break
        end
        for i in 1:soil.N sim_profile[t, i] = Float64(soil.θ[i]) end
    end
    
    if is_diverged continue end
    
    # 🌟 指标计算：精准过滤 4-10 月
    month_mask = [4 <= month(d) <= 10 for d in df_merged.date_only]

    layer_nses = Float64[]
    layer_r2s  = Float64[]
    
    for j in 1:8
        obs_layer = obs_matrix[:, j]
        sim_layer = sim_profile[:, j]
        
        valid_idx = findall(i -> !ismissing(obs_layer[i]) && !isnan(obs_layer[i]) && month_mask[i], 1:length(obs_layer))
        
        if length(valid_idx) < 10
            push!(layer_nses, NaN)
            push!(layer_r2s, NaN)
            continue
        end
        
        obs_valid = Float64.(obs_layer[valid_idx])
        sim_valid = sim_layer[valid_idx]
        
        var_obs = sum((obs_valid .- mean(obs_valid)).^2)
        nse_j = (var_obs == 0) ? NaN : 1 - (sum((obs_valid .- sim_valid).^2) / var_obs)
        push!(layer_nses, nse_j)
        
        r_val = (std(obs_valid) == 0 || std(sim_valid) == 0) ? NaN : cor(obs_valid, sim_valid)
        r2_j = isnan(r_val) ? NaN : r_val^2
        push!(layer_r2s, r2_j)
    end
    
    mean_nse = mean(filter(!isnan, layer_nses))
    mean_r2  = mean(filter(!isnan, layer_r2s))
    
    lock(io_lock) do
        site_metrics_records[SITE_NAME] = [layer_nses; mean_nse; layer_r2s; mean_r2]
        
        # 🌟 添加时间序列导出以便排查
        df_sim_export = DataFrame(
            Site = fill(SITE_NAME, nrow(df_merged)),
            Date = df_merged.date_only,
            PET_mm = coalesce.(df_merged.Daily_ET_mm, 0.0),
            Infiltration_mm = coalesce.(df_merged.Predicted_Infiltration, 0.0) 
        )
        for j in 1:8
            depth = depth_labels[j]
            df_sim_export[!, Symbol("Obs_SM_$(depth)cm")] = obs_matrix[:, j]
            df_sim_export[!, Symbol("Sim_SM_$(depth)cm")] = sim_profile[:, j]
        end
        push!(all_sim_dfs, df_sim_export)

        if SITE_NAME in plot_sites_list
            p_layers = plot(layout=(4, 2), size=(1300, 1100), margin=4Plots.mm)
            dates_to_plot = df_merged.date_only
            
            for j in 1:8
                depth = depth_labels[j]
                nse_val = isnan(layer_nses[j]) ? "NaN" : round(layer_nses[j], digits=3)
                r2_val  = isnan(layer_r2s[j])  ? "NaN" : round(layer_r2s[j], digits=3)
                
                plot!(p_layers[j], dates_to_plot, obs_matrix[:, j], label="Obs", color=:black, lw=1.5)
                plot!(p_layers[j], dates_to_plot, sim_profile[:, j], label="Sim(Dai_PET)", color=:forestgreen, lw=1.5, ls=:dash,
                      title="$(depth)cm | NSE: $nse_val | R²: $r2_val", ylabel="θ", legend=false, grid=true)
            end
            plot!(p_layers, plot_title="Site: $SITE_NAME | Mean 4-10m NSE: $(isnan(mean_nse) ? "NaN" : round(mean_nse, digits=3)) | Mean 4-10m R²: $(isnan(mean_r2) ? "NaN" : round(mean_r2, digits=3))")
            savefig(p_layers, joinpath(plot_dir, "Daily_SM_PET_Dai2019_$(SITE_NAME).png"))
        end
    end
end

# ====================================================================
# 6. 整理导出与报告
# ====================================================================
println("\n📊 [4/4] 正在整理并导出...")

if !isempty(all_sim_dfs)
    df_all_sim = vcat(all_sim_dfs...)
    path_all_sim = joinpath(output_dir, "All_Sites_Simulated_SM_TimeSeries.csv")
    CSV.write(path_all_sim, df_all_sim)
    println("✅ 所有站点的含水率时间序列已合并保存至: ", path_all_sim)
end

df_site_metrics = DataFrame(
    Site = String[],
    NSE_10cm = Float64[], NSE_20cm = Float64[], NSE_30cm = Float64[], NSE_40cm = Float64[],
    NSE_50cm = Float64[], NSE_60cm = Float64[], NSE_80cm = Float64[], NSE_100cm = Float64[], Mean_NSE = Float64[],
    R2_10cm = Float64[], R2_20cm = Float64[], R2_30cm = Float64[], R2_40cm = Float64[],
    R2_50cm = Float64[], R2_60cm = Float64[], R2_80cm = Float64[], R2_100cm = Float64[], Mean_R2 = Float64[]
)

for (site, metrics) in site_metrics_records
    push!(df_site_metrics, (site, 
                            metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7], metrics[8], metrics[9],
                            metrics[10], metrics[11], metrics[12], metrics[13], metrics[14], metrics[15], metrics[16], metrics[17], metrics[18]))
end

sort!(df_site_metrics, :Mean_R2, rev=true)

global_mean_nse = mean(filter(!isnan, df_site_metrics.Mean_NSE))
global_mean_r2  = mean(filter(!isnan, df_site_metrics.Mean_R2))

push!(df_site_metrics, ("GLOBAL_MEAN", 
                        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, global_mean_nse,
                        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, global_mean_r2))

CSV.write(joinpath(output_dir, "All_Sites_Metrics_PET_BEPS_Dai2019_Continuous_4to10Months.csv"), df_site_metrics)

report_path = joinpath(output_dir, "Global_Metrics_Report.txt")
open(report_path, "w") do io
    println(io, "=====================================================")
    println(io, "🏆 PET 驱动 + BEPS水势胁迫 + 戴院士参数 (连续时间轴修复版)")
    println(io, "=====================================================")
    println(io, "✅ 参与模拟有效站点总数 : ", nrow(df_site_metrics) - 1)
    println(io, "🚀 [4-10月] 全局所有层平均 NSE: ", isnan(global_mean_nse) ? "NaN" : round(global_mean_nse, digits=4))
    println(io, "🎯 [4-10月] 全局所有层平均 R² : ", isnan(global_mean_r2) ? "NaN" : round(global_mean_r2, digits=4))
end

println("\n=====================================================")
println("🏆 运行完成！时间黑洞已修复，每日蒸发照常进行。")
println("全局平均 NSE: $(isnan(global_mean_nse) ? "NaN" : round(global_mean_nse, digits=4))")
println("全局平均 R² : $(isnan(global_mean_r2) ? "NaN" : round(global_mean_r2, digits=4))")
println("=====================================================\n")