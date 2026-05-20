using Pkg
using NCDatasets, DataFrames, CSV, Dates, Statistics, StatsBase
using Plots
using Base.Threads

ENV["GKSwstype"] = "nul"
default(fontfamily="sans-serif")

# ====================================================================
# 1. 路径与核心物理阈值配置 (v8: 完整版)
# ====================================================================
file_sm     = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\sm\CHinaSM_hourly_2018-2019.nc"
output_dir  = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\Smart_Cleaned_Sites_v8/"
plot_dir    = joinpath(output_dir, "Smart_Passed_Plots")

if !isdir(output_dir) mkpath(output_dir) end
if !isdir(plot_dir) mkpath(plot_dir) end

# --- A. 基础质量阈值 ---
const MAX_LAYER_MISSING = 0.2     # 单层有效数据缺失 > 20% 剔除
const ABS_MAX_THETA     = 0.65     # 物理上限 (孔隙度极限)
const ABS_MIN_THETA     = 0.0      # 物理下限 
const MIN_VARIANCE      = 1e-6     # 全年方差绝对死线
const MAX_ZERO_RATIO    = 0.05     # 允许精确0.000值的最高比例

# --- B. 非物理突变阈值 ---
const MAX_FLICKERS      = 30       # 高频震荡(>0.15)次数上限 (条形码噪声)
const MAX_SUDDEN_DROPS  = 5        # 1小时内断崖下跌(<-0.10)次数上限 (方块台阶)

# --- C. 三段式停滞(卡死)检测阈值 ---
const THETA_SAT_THRESHOLD = 0.40   # 饱和高位界限
const THETA_LOW_THRESHOLD = 0.10   # 干旱低位界限
const STUCK_HOURS_SAT    = 336     # 高位饱和(>0.40): 停滞 > 14天 剔除 (地下水顶托/量程溢出)
const STUCK_HOURS_NORMAL = 17280     # 中位正常(0.10~0.40): 停滞 > 半年 剔除 (保护深层稳定)
const STUCK_HOURS_LOW    = 4320    # 低位冰冻(<0.10): 停滞 > 半年 剔除

# --- D. 优先流检测阈值 (新增) ---
const PREF_CORR_LIMIT    = 0.3     # 10cm与60cm入渗脉冲相关性上限

# ====================================================================
# 2. 核心去噪与诊断函数
# ====================================================================

function remove_spikes!(layer_data::Vector{Float64})
    # 1. 物理极值直接转 NaN
    for i in 1:length(layer_data)
        if !isnan(layer_data[i])
            if layer_data[i] > ABS_MAX_THETA || layer_data[i] < ABS_MIN_THETA
                layer_data[i] = NaN
            end
        end
    end
    # 2. 孤立电信号毛刺(Spikes)转 NaN
    for i in 2:(length(layer_data)-1)
        prev, curr, next = layer_data[i-1], layer_data[i], layer_data[i+1]
        if !isnan(prev) && !isnan(curr) && !isnan(next)
            if (curr - prev > 0.15) && (curr - next > 0.15)
                layer_data[i] = NaN
            elseif (prev - curr > 0.15) && (next - curr > 0.15)
                layer_data[i] = NaN
            end
        end
    end
end

function diagnose_site_smart(obs_matrix::Matrix{Float64})
    n_time, n_depth = size(obs_matrix)
    
    # === 阶段 1: 逐层质量检验 ===
    for d in 1:n_depth
        layer_data = obs_matrix[:, d]
        remove_spikes!(layer_data) # 先去噪
        
        valid_mask = .!isnan.(layer_data)
        valid_vals = layer_data[valid_mask]
        
        # 1. 缺失率与绝对死线
        if (1.0 - length(valid_vals)/n_time) > MAX_LAYER_MISSING
            return false, "Reject: 第$(d)层缺失或异常值过多(>20%)"
        end
        if var(valid_vals) < MIN_VARIANCE
            return false, "Reject: 第$(d)层数据方差过小(绝对死线)"
        end
        if count(x -> x <= 0.001, valid_vals) / length(valid_vals) > MAX_ZERO_RATIO
            return false, "Reject: 第$(d)层包含过多绝对零值(断线)"
        end
        
        # 2. 噪声与突变统计
        flickers, sudden_drops = 0, 0
        for i in 2:length(layer_data)
            if !isnan(layer_data[i]) && !isnan(layer_data[i-1])
                delta = layer_data[i] - layer_data[i-1]
                if abs(delta) > 0.15 flickers += 1 end
                if delta < -0.10 sudden_drops += 1 end
            end
        end
        if flickers > MAX_FLICKERS
            return false, "Reject: 第$(d)层存在高频剧烈震荡(条形码噪声)"
        end
        if sudden_drops > MAX_SUDDEN_DROPS
            return false, "Reject: 第$(d)层存在非物理断崖下跌(方块台阶)"
        end
        
        # 3. 三段式停滞卡死检测
        max_run, current_run, stuck_val = 0, 1, NaN
        for i in 2:length(layer_data)
            if !isnan(layer_data[i]) && layer_data[i] == layer_data[i-1]
                current_run += 1
            else
                if current_run > max_run
                    max_run = current_run
                    stuck_val = layer_data[i-1]
                end
                current_run = 1
            end
        end
        if !isnan(stuck_val)
            if stuck_val >= THETA_SAT_THRESHOLD && max_run > STUCK_HOURS_SAT
                return false, "Reject: 第$(d)层高位饱和卡死(>0.4且超14天)"
            elseif stuck_val >= THETA_LOW_THRESHOLD && stuck_val < THETA_SAT_THRESHOLD && max_run > STUCK_HOURS_NORMAL
                return false, "Reject: 第$(d)层中位异常停滞(超半年)"
            elseif stuck_val < THETA_LOW_THRESHOLD && max_run > STUCK_HOURS_LOW
                return false, "Reject: 第$(d)层低位死线停滞(超半年)"
            end
        end
    end
    
    # === 阶段 2: 剖面物理一致性检验 ===
    
    # 4. 浅层垂直逻辑 (防止传感器脱节或放反)
    valid_pairs = .!isnan.(obs_matrix[:, 1]) .& .!isnan.(obs_matrix[:, 2])
    if sum(valid_pairs) > 100
        c_val = cor(obs_matrix[valid_pairs, 1], obs_matrix[valid_pairs, 2])
        if isnan(c_val) || c_val < 0.05
            return false, "Reject: 10cm与20cm水文逻辑脱节(Corr<0.05)"
        end
    end

    # 5. 🌟 优先流/大孔隙流拦截 (1D基质流的保卫者)
    # 检测 10cm(层1) 与 60cm(层6) 的瞬时入渗脉冲相关性
    layer1, layer6 = obs_matrix[:, 1], obs_matrix[:, 6]
    valid_pref = .!isnan.(layer1) .& .!isnan.(layer6)
    valid_idx = findall(valid_pref)
    
    diff_1, diff_6 = Float64[], Float64[]
    for i in 2:length(valid_idx)
        # 确保时间步是连续的
        if valid_idx[i] == valid_idx[i-1] + 1
            d1 = layer1[valid_idx[i]] - layer1[valid_idx[i-1]]
            d6 = layer6[valid_idx[i]] - layer6[valid_idx[i-1]]
            # 仅在表层发生明显降雨/入渗 (>0.005) 时进行追踪
            if d1 > 0.005
                push!(diff_1, d1)
                push!(diff_6, d6)
            end
        end
    end
    
    # 如果有效降雨脉冲超过 10 次，且60cm与10cm的入渗突变高度同步
    if length(diff_1) > 10
        c_pref = cor(diff_1, diff_6)
        if !isnan(c_pref) && c_pref > PREF_CORR_LIMIT
            return false, "Reject: 存在严重优先流影响(10cm与60cm同频响应)"
        end
    end

    return true, "Passed: 高质量 1D 基质流站点"
end

# ====================================================================
# 3. 加载数据并执行清洗
# ====================================================================
println("📌 正在加载 NetCDF 数据并执行 v8 终极清洗...")
ds = NCDataset(file_sm, "r")
all_sites_str = string.(ds["site"][:])
dates_nc = DateTime.(ds["time"][:])
depth_cols = ["depth_10", "depth_20", "depth_30", "depth_40", "depth_50", "depth_60", "depth_80", "depth_100"]

data_dict = Dict{String, AbstractMatrix}()

if haskey(ds, "SM")
    sm_var = ds["SM"]
    dim_names = dimnames(sm_var)
    t_idx = findfirst(isequal("time"), dim_names)
    s_idx = findfirst(isequal("site"), dim_names)
    d_idx = findfirst(isequal("depth"), dim_names)
    raw_sm_perm = permutedims(sm_var[:, :, :], (t_idx, s_idx, d_idx))
    for (j, col) in enumerate(depth_cols)
        data_dict[col] = raw_sm_perm[:, :, j]
    end
else
    error("未在文件中找到 'SM' 变量！")
end
close(ds)

df_all = DataFrame(Site=String[], Status=String[], Reason=String[])
passed_sites = String[]

for (idx, site) in enumerate(all_sites_str)
    obs_matrix = zeros(Float64, length(dates_nc), 8)
    for (j, col) in enumerate(depth_cols)
        raw_seq = data_dict[col][:, idx]
        obs_matrix[:, j] .= [ismissing(x) ? NaN : (Float64(x) > 1.0 ? Float64(x)/100.0 : Float64(x)) for x in raw_seq]
    end
    
    is_passed, reason = diagnose_site_smart(obs_matrix)
    push!(df_all, (site, is_passed ? "Passed" : "Rejected", reason))
    if is_passed push!(passed_sites, site) end
end

CSV.write(joinpath(output_dir, "All_Sites_SmartCleaning_Report_v8.csv"), df_all)
df_passed = filter(row -> row.Status == "Passed", df_all)
CSV.write(joinpath(output_dir, "Passed_Sites_List_v8.csv"), df_passed)

# ====================================================================
# 4. 统计与多线程绘图
# ====================================================================
println("\n=====================================================")
println("📊 智能清洗结果总报告 (v8 终极版)")
println("=====================================================")
println("总站点数 \t\t: ", nrow(df_all))
println("✅ 通过清洗 (Passed)\t: ", length(passed_sites))
println("❌ 物理拒收 (Rejected)\t: ", nrow(df_all) - length(passed_sites))

println("\n[Reject 详细原因分类统计]")
reject_reasons = countmap(filter(r -> r.Status == "Rejected", df_all).Reason)
for (reason, count) in sort(collect(reject_reasons), by=x->x[2], rev=true)
    println(" - ", rpad(reason, 50, ' '), ": $count 个")
end
println("=====================================================\n")

println("🎨 正在为通过清洗的优质站点出图...")
io_lock = ReentrantLock()

Threads.@threads for site in passed_sites
    site_idx = findfirst(isequal(site), all_sites_str)
    obs_matrix = zeros(Float64, length(dates_nc), 8)
    for (j, col) in enumerate(depth_cols)
        raw_seq = data_dict[col][:, site_idx]
        obs_matrix[:, j] .= [ismissing(x) ? NaN : (Float64(x) > 1.0 ? Float64(x)/100.0 : Float64(x)) for x in raw_seq]
        remove_spikes!(obs_matrix[:, j])
    end
    
    lock(io_lock) do
        p = plot(layout=(4, 2), size=(1400, 1000), margin=5Plots.mm)
        depth_labels = [10, 20, 30, 40, 50, 60, 80, 100]
        for j in 1:8
            plot!(p[j], dates_nc, obs_matrix[:, j], label="Obs", color=:black, lw=1.2,
                  title="Depth: $(depth_labels[j])cm", ylabel="θ (m³/m³)", legend=(j==1 ? :topright : false), grid=true)
        end
        plot!(p, plot_title="Site: $site (v8: Matrix Flow Qualified)")
        savefig(p, joinpath(plot_dir, "Passed_Obs_$site.png"))
    end
end
println("🎉 全部任务完成！")
