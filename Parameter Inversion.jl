using Pkg
Pkg.activate("C:\\Users\\26332\\OneDrive\\Desktop\\kong mission\\mechin learning\\code\\infiltration\\SoilDiffEqs.jl-master")
Pkg.instantiate()

using SoilDifferentialEquations, Ipaper, Dates, CSV, DataFrames
using Base.Threads  
using Statistics 
using Plots 
using Random  
using NCDatasets   

ENV["GKSwstype"] = "nul"

# =========================================================
# 工具函数
# =========================================================
# 🌟 改进后的 NSE 计算函数，支持传入时间掩码 (time_mask)
function get_nse(sim, obs; time_mask=nothing)
    valid = .!isnan.(sim) .& .!isnan.(obs) .& .!ismissing.(obs)
    if time_mask !== nothing
        valid = valid .& time_mask
    end
    
    s = Float64.(sim[valid])
    o = Float64.(obs[valid])
    if length(s) < 2
        return NaN
    end
    mean_o = sum(o) / length(o)
    num = sum((s .- o).^2)
    den = sum((o .- mean_o).^2)
    return den == 0.0 ? NaN : round(1.0 - num/den, digits=3)
end

is_invalid(x) = ismissing(x) || isnan(x)

# 🌟 核心算法：剔除土壤水分数据的“毛刺”与异常跳变
function remove_spikes_and_outliers!(vec::AbstractVector{Float64})
    n = length(vec)
    vec_clean = copy(vec)
    
    valid_data = filter(!isnan, vec)
    if isempty(valid_data) return end
    
    is_percentage = mean(valid_data) > 1.5
    min_jump = is_percentage ? 5.0 : 0.05
    
    # 阶段 1：孤立尖峰剔除
    for i in 2:n-1
        if isnan(vec[i]) continue end
        
        prev_idx, next_idx = i - 1, i + 1
        while prev_idx > 1 && isnan(vec[prev_idx]) prev_idx -= 1 end
        while next_idx < n && isnan(vec[next_idx]) next_idx += 1 end
        
        if isnan(vec[prev_idx]) || isnan(vec[next_idx]) continue end
        
        diff1 = vec[i] - vec[prev_idx]
        diff2 = vec[i] - vec[next_idx]
        
        if sign(diff1) == sign(diff2) && abs(diff1) > min_jump && abs(diff2) > min_jump
            vec_clean[i] = NaN
        end
    end
    
    vec .= vec_clean
    
    # 阶段 2：Hampel 滤波剔除
    half_w = 12 
    for i in 1:n
        if isnan(vec[i]) continue end
        
        start_idx = max(1, i - half_w)
        end_idx = min(n, i + half_w)
        window_data = filter(!isnan, vec[start_idx:end_idx])
        
        if length(window_data) >= 5
            med = median(window_data)
            mad_val = median(abs.(window_data .- med))
            threshold = max(3.0 * 1.4826 * mad_val, min_jump)
            
            if abs(vec[i] - med) > threshold
                vec_clean[i] = NaN
            end
        end
    end
    
    vec .= vec_clean
end

# =========================================================
# 模型配置及输入
# =========================================================
config_file = "my_config.yaml"
my_out_dir = "C:\\Users\\26332\\OneDrive\\Desktop\\kong mission\\mechin learning\\code\\infiltration\\SoilDiffEqs.jl-master\\结果图\\Campbell-3hourly-NSE(重筛选4-10月-全图像)\\" 
mkpath(my_out_dir)
config = load_config(config_file)
(; zs_obs_orgin, zs_obs, scale_factor) = config

# =========================================================
# 🌟 读取指定参与反演的站点列表
# =========================================================
sites_file = "C:\\Users\\26332\\OneDrive\\Desktop\\kong mission\\mechin learning\\code\\infiltration\\Smart_Cleaned_Sites_v8\\Passed_Sites_List_v8.csv"
println("正在读取指定站点列表: $sites_file")
passed_sites_df = CSV.read(sites_file, DataFrame)
allowed_sites = Set(string.(passed_sites_df.Site))

# =========================================================
# 🌟 NetCDF 数据读取与严谨的维度对齐
# =========================================================
println("正在读取含水率 NetCDF 数据...")
file_sm = "C:\\Users\\26332\\OneDrive\\Desktop\\kong mission\\mechin learning\\code\\infiltration\\data\\sm\\CHinaSM_hourly_2018-2019.nc" 

ds = NCDataset(file_sm, "r")
all_sites_raw = ds["site"][:]  
dates_nc = ds["time"][:]       
dates_hourly = DateTime.(dates_nc) 

depth_cols =[
    "depth_10", "depth_20", "depth_30", "depth_40", 
    "depth_50", "depth_60", "depth_80", "depth_100"
]
# 用于输出参数时的物理深度标签
depth_labels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]

data_dict = Dict{String, Matrix{Union{Missing, Float64}}}()

let 
    if haskey(ds, depth_cols[1])
        for col in depth_cols
            dim_names = dimnames(ds[col])
            raw_data = ds[col][:, :]
            
            t_idx = findfirst(isequal("time"), dim_names)
            s_idx = findfirst(isequal("site"), dim_names)
            
            data_dict[col] = permutedims(raw_data, (t_idx, s_idx))
        end
    elseif haskey(ds, "SM") 
        sm_var = ds["SM"]
        dim_names = dimnames(sm_var)
        
        t_idx = findfirst(isequal("time"), dim_names)
        d_idx = findfirst(isequal("depth"), dim_names)
        s_idx = findfirst(isequal("site"), dim_names)
        
        raw_sm = sm_var[:, :, :]
        
        perm_order = (t_idx, s_idx, d_idx)
        raw_sm_perm = permutedims(raw_sm, perm_order)
        
        for (j, col) in enumerate(depth_cols)
            data_dict[col] = raw_sm_perm[:, :, j]
        end
    else
        error("无法识别 NC 文件变量，请检查文件内部是否为 depth_10 等。")
    end
end
close(ds) 

# ---------------------------------------------------------
# 仅提取目标站点并处理异常跳变（不填补缺失）
# ---------------------------------------------------------
println("正在提取指定站点的数据，并执行【毛刺剔除算法】（保留缺失状态）...")
valid_sites_data = Dict{String, Matrix{Float64}}()

for i in 1:length(all_sites_raw)
    site_name = string(all_sites_raw[i])
    
    if !(site_name in allowed_sites)
        continue
    end
    
    site_mat_raw = zeros(Union{Missing, Float64}, length(dates_hourly), length(depth_cols))
    for (j, col) in enumerate(depth_cols)
        site_mat_raw[:, j] .= data_dict[col][:, i]
    end
    
    site_mat_clean = zeros(Float64, size(site_mat_raw))
    
    for j in 1:length(depth_cols)
        for k in 1:length(dates_hourly)
            val = site_mat_raw[k, j]
            if ismissing(val) || isnan(val) || val <= 0.0
                site_mat_clean[k, j] = NaN
            else
                site_mat_clean[k, j] = Float64(val)
            end
        end
        
        layer_data = site_mat_clean[:, j]
        remove_spikes_and_outliers!(layer_data)
        site_mat_clean[:, j] .= layer_data
    end
    
    valid_sites_data[site_name] = site_mat_clean
end

target_sites = collect(keys(valid_sites_data))

println("\n" * "="^48)
println("📊 数据提取统计报告")
println("   ▶ 目标文件提供站点数 : $(length(allowed_sites))")
println("   ▶ NetCDF中成功匹配   : $(length(target_sites))")
println("   ✅ 毛刺与跳变已全部被置为 NaN，原缺失值保持不动")
println("="^48 * "\n")

if isempty(target_sites)
    error("未能从 NetCDF 中匹配到任何指定文件中的站点，请检查。")
end


# =========================================================
# 🌟 手动控制：指定要运行的站点范围 (方便按需分批执行)
# =========================================================
run_start_idx = 401       
run_end_idx   = 594  

run_start_idx = max(1, run_start_idx)
run_end_idx   = min(run_end_idx, length(target_sites))

if run_start_idx > run_end_idx
    error("起始索引大于结束索引，请检查范围设置！")
end

run_sites = target_sites[run_start_idx:run_end_idx]

num_plots = min(521, length(run_sites))
sites_to_plot = shuffle(run_sites)[1:num_plots]


# =========================================================
# 🌟 多线程反演 (仅针对当前指定的范围)
# =========================================================
const thread_lock = ReentrantLock()

all_nses = Float64[]   

sim_file = joinpath(my_out_dir, "All_Sites_SimData_Merged.csv")
param_file = joinpath(my_out_dir, "All_Sites_Params_Merged.csv")

println("\n🚀 启动多线程计算！当前可用 CPU 线程数: ", nthreads())
println("📦 当前设定运行索引:[$run_start_idx 到 $run_end_idx]，共提取 $(length(run_sites)) 个站点。")
println("⏱️ 时间尺度: 抽样为 3 小时步长(1:3:end)。")
println("🔥 预热期设定: 【已取消预热期】，从第1天起计算误差。")
println("🍁 目标函数设定: 优化器与出图【仅考虑 4-10 月】的数据计算误差。")
println("🌟 无差别输出: 将输出【所有跑通站点】的参数与完整的含水率时序数据。")
println("========================================================")

const original_stdout = stdout
redirect_stdout(devnull) 

# 生成 3 小时时间轴
dates_3hourly = dates_hourly[1:3:end]

batch_sim_dfs = DataFrame[]
batch_param_dfs = DataFrame[]

Threads.@threads for i in 1:length(run_sites)
    SITE_NAME = run_sites[i]
    t_id = threadid() 
    
    try
        site_config = deepcopy(config) 
        
        data_origin_3hourly = valid_sites_data[SITE_NAME][1:3:end, :]
        data_obs_full = interp_data_depths(data_origin_3hourly .* scale_factor, zs_obs_orgin, zs_obs)
        
        # 寻找【所有深度】都有数据的完美初始点
        start_idx = nothing
        n_rows, n_cols = size(data_obs_full)
        
        for t in 1:n_rows
            all_valid = true
            for c in 1:n_cols
                if isnan(data_obs_full[t, c]) || ismissing(data_obs_full[t, c])
                    all_valid = false
                    break
                end
            end
            
            if all_valid
                start_idx = t
                break
            end
        end
        
        if start_idx !== nothing
            data_obs = data_obs_full[start_idx:end, :]
            dates_sliced = dates_3hourly[start_idx:end]
        else
            fallback_idx = findfirst(x -> !isnan(x) && !ismissing(x), data_obs_full[:, 1])
            start_idx = fallback_idx !== nothing ? fallback_idx : 1
            data_obs = data_obs_full[start_idx:end, :]
            dates_sliced = dates_3hourly[start_idx:end]
            
            for col_idx in 1:size(data_obs, 2)
                if isnan(data_obs[1, col_idx]) || ismissing(data_obs[1, col_idx])
                    valid_idx = findfirst(x -> !isnan(x) && !ismissing(x), data_obs[:, col_idx])
                    if valid_idx !== nothing
                        data_obs[1, col_idx] = data_obs[valid_idx, col_idx]
                    else
                        data_obs[1, col_idx] = 0.30 
                    end
                end
            end
        end

        # ====================================================
        # 🌟 核心修改：生成当前站点的 4-10 月时间掩码，并替换优化目标函数
        # ====================================================
        month_arr = month.(dates_sliced)
        valid_months_mask = (month_arr .>= 4) .& (month_arr .<= 10)

        function masked_opt_nse(obs, sim)
            # 过滤 NaN、Missing 以及 非 4-10 月的数据
            valid = .!isnan.(sim) .& .!isnan.(obs) .& .!ismissing.(obs) .& valid_months_mask
            s = Float64.(sim[valid])
            o = Float64.(obs[valid])
            
            # 如果有效数据太少，返回一个极差的惩罚值（防止优化器崩溃）
            if length(s) < 10
                return -999.0 
            end
            
            mean_o = mean(o)
            den = sum((o .- mean_o).^2)
            return den == 0.0 ? -999.0 : 1.0 - sum((s .- o).^2)/den
        end

        # 覆盖 Config 中的默认目标函数
        site_config.of_fun = masked_opt_nse

        final_nse_ref = Ref{Float64}(NaN) 
        sim_data_ref = Ref{Matrix{Float64}}() 

        # ====================================================
        # 🌟 核心修改：在出图与最终评价阶段应用 4-10 月的掩码 (移除预热期约束)
        # ====================================================
        custom_plot_fun = (; ysim, yobs, dates, depths, fout) -> begin
            # 严格时间掩码：月份在 4-10 之间
            eval_mask = (month.(dates) .>= 4) .& (month.(dates) .<= 10)
            
            # 计算仅包含非冻结期(4-10月)的有效 NSE 分数
            current_nse = get_nse(ysim, yobs; time_mask=eval_mask)
            
            final_nse_ref[] = current_nse 
            sim_data_ref[] = copy(ysim) 
            
            if SITE_NAME in sites_to_plot
                new_fout = joinpath(my_out_dir, "Site_$(SITE_NAME)_Result.png")
                lock(thread_lock) do 
                    n_layers = size(yobs, 2)
                    plts =[]
                    for j in 1:n_layers
                        # 图形本身画出完整的 2 年曲线，帮助您检视跨年期间的模型连续性行为
                        p_sub = plot(dates, yobs[:, j], label="Obs", color=:black, linewidth=1.5)
                        plot!(p_sub, dates, ysim[:, j], label="Sim", color=:red, linewidth=1.5, linestyle=:dash)
                        
                        plot!(p_sub, title="Depth: $(depths[j])", titlefontsize=9, legend=(j==1 ? :topright : false))
                        push!(plts, p_sub)
                    end
                    p_all = plot(plts..., layout=(n_layers, 1), size=(1000, 150 * n_layers),
                                 plot_title="Site: $(SITE_NAME)  |  Apr-Oct NSE: $(current_nse)",
                                 plot_titlefontsize=14, margin=3Plots.mm)
                    savefig(p_all, new_fout) 
                end
            end
        end

        model_out = Soil_main(
            site_config,            
            data_obs,               
            string(SITE_NAME), 
            dates_sliced;          
            maxn=site_config.maxn,  
            plot_fun=custom_plot_fun,  
            plot_initial=false,          
            method_retention="Campbell"  
        )
        
        nse_val = final_nse_ref[]
        
        lock(thread_lock) do
            if !isnan(nse_val)
                push!(all_nses, nse_val) 
            end

            # ====================================================
            # 🌟 核心提取一：无差别提取站点的所有 8 层参数
            # ====================================================
            try
                soil_obj = model_out isa Tuple ? model_out[1] : model_out
                par = soil_obj.param
                
                n_par_layers = length(par.Ksat)
                n_extract = min(length(depth_labels), n_par_layers)
                
                for l in 1:n_extract
                    b_val = hasproperty(par, :b) ? par.b[l] : (hasproperty(par, :B) ? par.B[l] : NaN)
                    
                    psi_e_val = if hasproperty(par, :ψ_sat)
                        par.ψ_sat[l]
                    elseif hasproperty(par, :ψ_ae)
                        par.ψ_ae[l]
                    elseif hasproperty(par, :ψ_e)
                        par.ψ_e[l]
                    elseif hasproperty(par, :he)
                        par.he[l]
                    else
                        NaN
                    end

                    param_df = DataFrame(
                        site      = SITE_NAME,
                        depth     = depth_labels[l],
                        Ksat      = par.Ksat[l],   
                        theta_s   = par.θ_sat[l],
                        b         = b_val,
                        psi_e     = psi_e_val,
                        NSE_Score = nse_val  # 保留 NSE 分数信息，无论高低
                    )
                    
                    if hasproperty(par, :θ_res)
                        insertcols!(param_df, :theta_r => par.θ_res[l])
                    end

                    push!(batch_param_dfs, param_df)
                end
            catch e_param
                println(original_stdout, "[线程 $t_id] ⚠️ 站点 $SITE_NAME 参数提取异常...")
            end

            # ====================================================
            # 🌟 核心提取二：无差别提取站点的完整时序数据矩阵
            # ====================================================
            if !isnan(nse_val)
                layer_10_obs = data_obs[:, 1]
                combined_sim_data = hcat(layer_10_obs, sim_data_ref[])
                
                col_names =[
                    "Depth_10", "Depth_20", "Depth_30", "Depth_40", 
                    "Depth_50", "Depth_60", "Depth_80", "Depth_100"
                ]
                
                sim_df = DataFrame(combined_sim_data, col_names)
                insertcols!(sim_df, 1, :site => SITE_NAME)
                insertcols!(sim_df, 2, :time => dates_sliced)
                
                push!(batch_sim_dfs, sim_df)
            end
        end
        
    catch e
        lock(thread_lock) do
            println(original_stdout, "[线程 $t_id] ❌ 站点 $SITE_NAME 异常: ", e)
        end
    end
end

lock(thread_lock) do
    if !isempty(batch_param_dfs)
        all_param_merged = vcat(batch_param_dfs...)
        CSV.write(param_file, all_param_merged, append=isfile(param_file))
        println(original_stdout, "\n💾[参数落盘] 成功写入 $(length(batch_param_dfs) ÷ 8) 个站点的全层物理参数！")
    end
    
    if !isempty(batch_sim_dfs)
        all_sim_merged = vcat(batch_sim_dfs...)
        CSV.write(sim_file, all_sim_merged, append=isfile(sim_file))
        println(original_stdout, "💾[模拟落盘] 成功将所有跑通站点的含水率时序数据【追加】到文件中。")
    else
        println(original_stdout, "⚠️ [模拟落盘] 本次索引无跑通站点，未产生时序数据追加。")
    end
end

redirect_stdout(original_stdout)

# =========================================================
# 🌟 本次运行结果报告
# =========================================================
if !isempty(all_nses)
    avg_all_nse = round(mean(all_nses), digits=3)
    
    println("\n" * "="^56)
    println("📊 本次运行统计 (索引: $run_start_idx 到 $run_end_idx)")
    println("   ▶ 提交运行站点数     : $(length(run_sites))")
    println("   ▶ 成功输出数据的站点数: $(length(all_nses))")
    println("   ▶ 本次跑通【平均 NSE】: $avg_all_nse (无预热期，已排除 11-3 月冻结期)")
    println("="^56 * "\n")
end
