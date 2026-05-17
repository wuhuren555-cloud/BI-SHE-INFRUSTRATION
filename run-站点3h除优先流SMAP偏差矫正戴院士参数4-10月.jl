using Pkg
Pkg.activate(raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\SoilDiffEqs.jl-master")
Pkg.instantiate()

using SoilDifferentialEquations, Ipaper
using DataFrames, CSV, Dates, XLSX 
using Plots, Statistics, Random, StatsBase
using NCDatasets 
using Base.Threads 

# ====================================================================
# 1. 路径配置
# ====================================================================
ENV["GKSwstype"] = "nul"
Random.seed!(42)
default(fontfamily="sans-serif") 

path_smap       = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\SMAP_Corrected_TargetSites.csv"
path_dai2019    = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\Dai2019_SoilProperties_sp2702.xlsx"
file_sm         = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\sm\CHinaSM_hourly_2018-2019.nc"
path_old_params = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\SoilDiffEqs.jl-master\结果图\Campbell-3hourly-NSE(重筛选4-10月-全图像)\All_Sites_Params_Merged.csv"
config_file     = "my_config_smap.yaml"

output_dir      = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\Physics_Forward_Dai2019_EvalAprOct/"
if !isdir(output_dir) mkpath(output_dir) end

plot_dir = joinpath(output_dir, "All_Successful_Sites_Plots")
if !isdir(plot_dir) mkpath(plot_dir) end

# ====================================================================
# 2. 读取数据
# ====================================================================
println("📌 [1/4] 正在加载 数据集 与 戴院士物理参数表...")

df_old_params = CSV.read(path_old_params, DataFrame)
allowed_sites = unique(string.(df_old_params.site))
df_smap = CSV.read(path_smap, DataFrame)

xf = XLSX.readxlsx(path_dai2019)
df_Ksat = DataFrame(XLSX.gettable(xf["Ksat"])); df_Ksat.site = string.(df_Ksat.site)
df_θsat = DataFrame(XLSX.gettable(xf["θsat"])); df_θsat.site = string.(df_θsat.site)
df_λ    = DataFrame(XLSX.gettable(xf["λ"]));    df_λ.site    = string.(df_λ.site)
df_ψsat = DataFrame(XLSX.gettable(xf["ψsat"])); df_ψsat.site = string.(df_ψsat.site)

excel_depths = [2.5, 10.0, 22.5, 45.0, 80.0, 150.0] 
excel_cols = ["l1", "l2", "l3", "l4", "l5", "l6"]

ds = NCDataset(file_sm, "r")
all_sites_raw = string.(ds["site"][:])  
dates_nc = ds["time"][:]       
dates_hourly = convert.(DateTime, dates_nc)

depth_cols =[
    "depth_10", "depth_20", "depth_30", "depth_40", 
    "depth_50", "depth_60", "depth_80", "depth_100"
]
depth_labels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]

data_dict = Dict{String, Matrix{Union{Missing, Float64}}}()
let 
    if haskey(ds, depth_cols[1])
        for col in depth_cols
            dim_names = dimnames(ds[col])
            data_dict[col] = permutedims(ds[col][:, :], (findfirst(isequal("time"), dim_names), findfirst(isequal("site"), dim_names)))
        end
    elseif haskey(ds, "SM") 
        sm_var = ds["SM"]
        dim_names = dimnames(sm_var)
        raw_sm_perm = permutedims(sm_var[:, :, :], (findfirst(isequal("time"), dim_names), findfirst(isequal("site"), dim_names), findfirst(isequal("depth"), dim_names)))
        for (j, col) in enumerate(depth_cols) data_dict[col] = raw_sm_perm[:, :, j] end
    end
end
close(ds)

valid_sites = intersect(unique(df_smap.site), all_sites_raw, unique(df_Ksat.site), allowed_sites)
total_sites = length(valid_sites)
site_metrics_records = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Float64, Float64}}()

# ====================================================================
# 3. 核心计算 
# ====================================================================
println("⚙️ [2/4] 启动 戴院士物理正演 (纯正参数注入 + 5分钟微积分装甲)...")
forward_config = load_config(config_file)
lk = ReentrantLock()

Threads.@threads for idx in 1:total_sites
    SITE_NAME = valid_sites[idx] 
    
    try 
        df_site_smap = filter(row -> string(row.site) == SITE_NAME, df_smap)
        sort!(df_site_smap, :datetime)
        
        s_idx = findfirst(isequal(SITE_NAME), all_sites_raw)
        df_obs_hourly = DataFrame(time = dates_hourly)
        for col in depth_cols df_obs_hourly[!, col] = data_dict[col][:, s_idx] end
        
        df_obs_aligned = DataFrame([col => Union{Missing, Float64}[] for col in depth_cols])
        for smap_t in df_site_smap.datetime
            diffs = abs.(df_obs_hourly.time .- smap_t)
            idx_closest = argmin(diffs)
            push!(df_obs_aligned, diffs[idx_closest] <= Millisecond(Hour(2)) ? df_obs_hourly[idx_closest, depth_cols] : [missing for _ in depth_cols])
        end
        for col in depth_cols df_obs_aligned[!, col] = [ismissing(x) ? missing : Float64(x)/100.0 for x in df_obs_aligned[!, col]] end
        
        df_merged = hcat(df_site_smap, df_obs_aligned)
        if nrow(df_merged) < 10 continue end 
        
        obs_matrix = Matrix(df_merged[:, depth_cols])
        dummy_obs = zeros(Float64, nrow(df_merged), forward_config.grid.N) 
        soil, _, _, _ = SoilDifferentialEquations.setup(forward_config, dummy_obs)
        
        # 🛡️ 稳压核心 1：强制微积分步长设为 5 分钟 (300秒)！防止高渗沙土炸毁矩阵！
        soil.dt = 300.0 
        
        row_K = df_Ksat[df_Ksat.site .== SITE_NAME, :]
        row_θ = df_θsat[df_θsat.site .== SITE_NAME, :]
        row_λ = df_λ[df_λ.site .== SITE_NAME, :]
        row_ψ = df_ψsat[df_ψsat.site .== SITE_NAME, :]
        
        last_K, last_θ, last_λ, last_ψ = NaN, NaN, NaN, NaN
        
        for i in 1:soil.N
            z_node = abs(soil.z[i] * 100.0) 
            col_name = excel_cols[argmin(abs.(excel_depths .- z_node))]
            
            val_K, val_θ, val_λ, val_ψ = row_K[1, col_name], row_θ[1, col_name], row_λ[1, col_name], row_ψ[1, col_name]
            
            if ismissing(val_K) || ismissing(val_θ) || ismissing(val_λ) || ismissing(val_ψ)
                if i != 1
                    soil.param.Ksat[i], soil.param.θ_sat[i], soil.param.b[i], ψsat_val = last_K, last_θ, last_λ, last_ψ
                end
            else
                raw_K = Float64(val_K)
                if raw_K < 0.0      last_K = (10^raw_K) * 360.0 
                elseif raw_K < 2.0  last_K = raw_K * 360.0      
                else                last_K = raw_K / 24.0 end
                
                # 🛡️ 稳压核心 2：物理限速器，防止极个别脏数据导致 Ksat 大于 30 毁坏矩阵
                last_K = clamp(last_K, 0.001, 30.0)
                
                idx_obs = argmin(abs.(depth_labels .- z_node))
                obs_layer_data = obs_matrix[:, idx_obs]
                valid_vals = filter(x -> !ismissing(x) && !isnan(x), obs_layer_data)
                real_max_theta = isempty(valid_vals) ? Float64(val_θ) : maximum(valid_vals)
                last_θ = max(Float64(val_θ), real_max_theta + 0.01)
                
                raw_λ = Float64(val_λ)
                last_λ = raw_λ > 1.0 ? raw_λ : (1.0 / raw_λ)
                raw_ψ = -abs(Float64(val_ψ))
                last_ψ = abs(raw_ψ) > 50.0 ? raw_ψ / 10.0 : raw_ψ
                
                soil.param.Ksat[i]  = last_K
                soil.param.θ_sat[i] = last_θ
                soil.param.b[i]     = last_λ
                ψsat_val            = last_ψ
            end
            
            if hasproperty(soil.param, :ψ_e) soil.param.ψ_e[i] = ψsat_val
            elseif hasproperty(soil.param, :ψ_sat) soil.param.ψ_sat[i] = ψsat_val end
        end

        # 🌟🌟🌟 破案神仙代码：强制刷新引擎内部矩阵！🌟🌟🌟
        SoilDifferentialEquations.Update_SoilParam_Param!(soil.param)

        for i in 1:soil.N
            z_node = abs(soil.z[i] * 100.0)
            idx_obs = argmin(abs.(depth_labels .- z_node))
            init_val = ismissing(obs_matrix[1, idx_obs]) ? (soil.param.θ_sat[i] * 0.5) : Float64(obs_matrix[1, idx_obs])
            soil.θ[i] = clamp(init_val, 0.02, soil.param.θ_sat[i] - 0.001)
            soil.θ_prev[i] = soil.θ[i]
        end
        
        # 强制底层用新的参数重新计算水力传导基础矩阵
        try 
            SoilDifferentialEquations.cal_ψ!(soil, soil.θ)
            SoilDifferentialEquations.cal_K!(soil, soil.θ) 
        catch; end
        
        sim_profile = zeros(Float64, nrow(df_merged), soil.N)
        for i in 1:soil.N sim_profile[1, i] = Float64(soil.θ[i]) end
        sink = zeros(Float64, soil.N) 
        
        for t in 2:nrow(df_merged)
            smap_sm = df_merged.sm_surface_corr[t]
            time_diff_hours = clamp((df_merged.datetime[t] - df_merged.datetime[t-1]).value / 3600000.0, 3.0, 24.0)
            
            # 因为 soil.dt 现在是 300秒，这里的步数会自动翻倍到 36 步，确保极度平滑的运算！
            steps_per_interval = max(1, round(Int, time_diff_hours / (soil.dt / 3600.0)))
            
            try
                if ismissing(smap_sm) || isnan(smap_sm) smap_sm = soil.θ[1] end
                smap_sm = clamp(smap_sm, 0.02, soil.param.θ_sat[1] - 0.001)
                ψ0 = SoilDifferentialEquations.Init_ψ0(soil, smap_sm)
                
                for _ in 1:steps_per_interval
                    SoilDifferentialEquations.soil_moisture!(soil, sink, ψ0; debug=false)
                    
                    # 🛡️ 稳压核心 3：绝对 NaN 感染阻断器！
                    if any(isnan.(soil.θ)) || any(isinf.(soil.θ))
                        error("NaN Detected") # 直接触发 catch 回滚，不让脏血进入矩阵
                    end
                    
                    for i in 1:soil.N soil.θ[i] = clamp(soil.θ[i], 0.015, soil.param.θ_sat[i] - 0.0001) end
                    SoilDifferentialEquations.cal_ψ!(soil, soil.θ)
                    for i in 1:soil.N soil.θ_prev[i] = soil.θ[i] end
                end
            catch e
                for i in 1:soil.N soil.θ[i] = soil.θ_prev[i] end
                try SoilDifferentialEquations.cal_ψ!(soil, soil.θ) catch; end
            end
            for i in 1:soil.N sim_profile[t, i] = Float64(soil.θ[i]) end
        end
        
        if any(isnan.(sim_profile)) || any(isinf.(sim_profile)) continue end

        # ---------------------------------------------------------
        # 🌟 算分逻辑：4-10 月评价掩码
        # ---------------------------------------------------------
        layer_nses, layer_r2s = Float64[], Float64[]
        months_arr = month.(df_merged.datetime)
        
        for j in 1:8
            obs_layer = obs_matrix[:, j]
            sim_layer = sim_profile[:, argmin(abs.(abs.(soil.z .* 100.0) .- depth_labels[j]))]
            
            valid_idx = findall(x -> !ismissing(obs_layer[x]) && !isnan(obs_layer[x]) && (4 <= months_arr[x] <= 10), 1:length(obs_layer))
            
            if length(valid_idx) < 10 
                push!(layer_nses, NaN); push!(layer_r2s, NaN); continue 
            end
            
            ov, sv = Float64.(obs_layer[valid_idx]), sim_layer[valid_idx]
            v_obs = sum((ov .- mean(ov)).^2)
            push!(layer_nses, (v_obs == 0) ? NaN : 1 - (sum((ov .- sv).^2) / v_obs))
            push!(layer_r2s, (std(ov) > 1e-7 && std(sv) > 1e-7) ? cor(ov, sv)^2 : NaN)
        end
        
        v_nses, v_r2s = filter(!isnan, layer_nses), filter(!isnan, layer_r2s)
        if isempty(v_nses) || isnan(mean(v_nses)) continue end
        m_nse, m_r2 = mean(v_nses), isempty(v_r2s) ? NaN : mean(v_r2s)
        
        lock(lk) do
            site_metrics_records[SITE_NAME] = (layer_nses, layer_r2s, m_nse, m_r2)
            
            p_layers = plot(layout=(4, 2), size=(1200, 1000), margin=4Plots.mm)
            dates_to_plot = df_merged.datetime
            for j in 1:8
                depth = depth_labels[j]
                nv_s = isnan(layer_nses[j]) ? "NaN" : round(layer_nses[j], digits=3)
                rv_s = isnan(layer_r2s[j]) ? "NaN" : round(layer_r2s[j], digits=3)
                sim_idx = argmin(abs.(abs.(soil.z .* 100.0) .- depth_labels[j]))
                
                plot!(p_layers[j], dates_to_plot, obs_matrix[:, j], label="Obs", color=:black, lw=1.2)
                plot!(p_layers[j], dates_to_plot, sim_profile[:, sim_idx], label="Sim(Dai2019)", color=:red, lw=1.2, ls=:dash,
                      title="$(depth)cm | NSE(4-10M):$nv_s | R²(4-10M):$rv_s", ylabel="θ", legend=false, grid=true)
            end
            plot!(p_layers, plot_title="Site: $SITE_NAME (Dai2019) | Mean NSE (Apr-Oct): $(round(m_nse, digits=3)) | R²: $(round(m_r2, digits=3))")
            savefig(p_layers, joinpath(plot_dir, "Dai2019_EvalAprOct_$(SITE_NAME).png"))
        end
    catch global_e end
end

# ====================================================================
# 5. 导出结果 (🌟 完全扩展至包含所有深度 R2)
# ====================================================================
println("\n📊 [4/4] 正在整理并导出...")
df_site_metrics = DataFrame(
    Site = String[], 
    NSE_10cm = Float64[], NSE_20cm = Float64[], NSE_30cm = Float64[], NSE_40cm = Float64[],
    NSE_50cm = Float64[], NSE_60cm = Float64[], NSE_80cm = Float64[], NSE_100cm = Float64[], 
    Mean_NSE = Float64[], 
    R2_10cm  = Float64[], R2_20cm  = Float64[], R2_30cm  = Float64[], R2_40cm  = Float64[],
    R2_50cm  = Float64[], R2_60cm  = Float64[], R2_80cm  = Float64[], R2_100cm = Float64[],
    Mean_R2  = Float64[]
)

for (site, m) in site_metrics_records
    ln, lr, mn, mr = m
    # 将层级NSE，平均NSE，层级R2，平均R2依次推入DataFrame
    push!(df_site_metrics, (
        site, 
        ln[1], ln[2], ln[3], ln[4], ln[5], ln[6], ln[7], ln[8], 
        mn, 
        lr[1], lr[2], lr[3], lr[4], lr[5], lr[6], lr[7], lr[8],
        mr
    ))
end

sort!(df_site_metrics, :Mean_NSE, rev=true)
CSV.write(joinpath(output_dir, "All_Sites_Dai2019_EvalAprOct_Metrics.csv"), df_site_metrics)

gn = mean(filter(!isnan, df_site_metrics.Mean_NSE))
gr = mean(filter(!isnan, df_site_metrics.Mean_R2))

println("\n=====================================================")
println("🏆 戴院士参数：连续推演 + 4-10月定向评估 报告完毕")
println("=====================================================")
println("✅ 成功生成图纸的站点数 : ", nrow(df_site_metrics))
println("🚀 全局平均 NSE (仅算4-10月): $(isnan(gn) ? "NaN" : round(gn, digits=4))")
println("📈 全局平均 R²  (仅算4-10月): $(isnan(gr) ? "NaN" : round(gr, digits=4))")
println("=====================================================\n")