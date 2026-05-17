using CSV
using DataFrames
using Dates
using Statistics
using LinearAlgebra  # 用于极速、稳定的矩阵运算求 VIF
using StatsBase
using Plots
using StatsPlots
using DecisionTree
using Random
using GLM
using NCDatasets     # 用于读取 NetCDF 文件

# 统一字体，防止中文/特殊符号乱码
default(fontfamily="Helvetica", fmt=:png, dpi=300)

# ====================================================================
# 0. 基础路径与智能时间解析器
# ====================================================================
path_daily_rain   = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\rainfall\prcp_CHM_PRE_V2_ChinaSM_2018-2019_sp2702.csv"
path_dynamic      = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Cleaned_Dynamic_Hourly_2018_2019.csv"
path_static       = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Cleaned_Static_Factors.csv"  

# 📌 更新：下渗数据新路径
path_infiltration = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\SoilDiffEqs.jl-master\结果图\Campbell-3hourly-NSE(重筛选4-10月-全图像)\3Hourly_Infiltration_Calculated_DailyScaled.csv" 

path_vpd_lai      = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Merged_VPD_LAI_Daily_Filtered.csv"
path_twi          = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Static_TWI_Feature.csv"
path_soil_params  = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\SoilDiffEqs.jl-master\结果图\Campbell-3hourly-NSE(重筛选4-10月-全图像)\All_Sites_Params_Merged.csv"
path_sm_nc        = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\sm\CHinaSM_hourly_2018-2019.nc" 

# 结果输出路径
output_dir        = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\Analysis_Results_4-10月除优先流预测结果/"
if !isdir(output_dir) mkpath(output_dir) end

# 智能解析时间，截取前19位并替换 "T" 为空格，兼容 "2018-03-17T15:00:00.0" 这种格式
parse_dt(x::AbstractString) = DateTime(replace(first(x, 19), "T" => " "), "yyyy-mm-dd HH:MM:SS")
parse_dt(x::DateTime) = x
parse_dt(x::Date) = DateTime(x)
parse_dt(x::Missing) = missing

parse_date(x::AbstractString) = Date(first(x, 10))
parse_date(x::DateTime) = Date(x)
parse_date(x::Date) = x
parse_date(x::Missing) = missing

# 辅助函数：获取第一笔非空数据作为当天的初始状态
function first_valid(x)
    for val in x
        if !ismissing(val) && !isnan(val)
            return val
        end
    end
    return missing
end

# ====================================================================
# 1. 核心升级：读取并拼接所有多源特征
# ====================================================================
println("📌[1/2] 正在加载并融合大一统物理特征...")

# --- A. 降水及降水历史特征 (前3天与前7天累计降水) ---
df_rain = CSV.read(path_daily_rain, DataFrame, types=Dict(:site => String))
df_rain.time = parse_date.(df_rain.time)
sort!(df_rain, [:site, :time])

df_rain_features = combine(groupby(df_rain, :site)) do group
    df_out = DataFrame(date_only = group.time)
    df_out.Prcp_Current_day = group.prec  
    df_out.Prcp_Lag1_day = [missing; group.prec[1:end-1]]
    df_out.Prcp_Lag2_day = [missing; missing; group.prec[1:end-2]]
    
    past_3d = Vector{Union{Missing, Float64}}(missing, nrow(group))
    for i in 4:nrow(group) past_3d[i] = sum(skipmissing(group.prec[i-3:i-1])) end
    df_out.Prcp_Past_3d_sum = past_3d

    past_7d = Vector{Union{Missing, Float64}}(missing, nrow(group))
    for i in 8:nrow(group) past_7d[i] = sum(skipmissing(group.prec[i-7:i-1])) end
    df_out.Prcp_Past_7d_sum = past_7d
    
    return df_out
end

# --- B1. 从 NetCDF 提取每日初始土壤含水率 (一二层) ---
println("📡 正在解析 NetCDF 土壤含水率数据...")
ds = NCDataset(path_sm_nc, "r")
all_sites_raw = String.(ds["site"][:])  
dates_nc = ds["time"][:]       
dates_hourly = DateTime.(dates_nc) 

depth_cols = [
    "depth_10", "depth_20", "depth_30", "depth_40", 
    "depth_50", "depth_60", "depth_80", "depth_100"
]
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

num_times = length(dates_hourly)
num_sites = length(all_sites_raw)

df_sm_nc = DataFrame(
    time = repeat(dates_hourly, outer=num_sites),
    site = repeat(all_sites_raw, inner=num_times),
    Soil_Water_L1_raw = vec(data_dict["depth_10"]),
    Soil_Water_L2_raw = vec(data_dict["depth_20"])
)
df_sm_nc.date_only = Date.(df_sm_nc.time)

sort!(df_sm_nc, [:site, :time])
df_sm_daily = combine(groupby(df_sm_nc, [:site, :date_only])) do group
    return DataFrame(
        Soil_Water_L1_initial = first_valid(group.Soil_Water_L1_raw),
        Soil_Water_L2_initial = first_valid(group.Soil_Water_L2_raw)
    )
end

# --- B2. 提取温度与雨强特征 ---
df_dyn = CSV.read(path_dynamic, DataFrame, types=Dict(:site => String))
df_dyn.datetime = parse_dt.(df_dyn.datetime)
sort!(df_dyn, [:site, :datetime]) 
df_dyn.date_only = Date.(df_dyn.datetime)

df_dyn_daily = combine(groupby(df_dyn,[:site, :date_only])) do group
    temp = first(group.Temperature_C)
    era5_hourly_rain = coalesce.(group.Precipitation_mm, 0.0)
    era5_daily_sum = sum(era5_hourly_rain)
    era5_max_hourly = maximum(era5_hourly_rain)
    peak_ratio = era5_daily_sum > 0.0 ? (era5_max_hourly / era5_daily_sum) : 0.0
    return DataFrame(Temp_C_initial = temp, ERA5_Peak_Ratio = peak_ratio)
end

# --- C. 目标变量：下渗量 ---
df_target = CSV.read(path_infiltration, DataFrame, types=Dict(:Site_ID => String))
rename!(df_target, :Site_ID => :site, :DateTime => :time_3h)
df_target.time_3h = parse_dt.(df_target.time_3h)
df_target.date_only = Date.(df_target.time_3h)

# 将此处的 :Inf_3h_Sum 改为 :Inf_3h_Final
df_target_daily = combine(groupby(df_target, [:site, :date_only]), :Inf_3h_Final => (x -> sum(skipmissing(x))) => :Inf_daily_Sum)
# --- D. VPD 与 LAI ---
df_vpd_lai = CSV.read(path_vpd_lai, DataFrame, types=Dict(:site => String))
df_vpd_lai.date_only = parse_date.(df_vpd_lai.date_only)
select!(df_vpd_lai,[:site, :date_only, :VPD, :LAI])

# --- E. TWI ---
df_twi = CSV.read(path_twi, DataFrame, types=Dict(:site => String))
select!(df_twi, [:site, :TWI])

# --- F. 土壤水力学参数 (Ksat, theta_s) ---
df_soil_params = CSV.read(path_soil_params, DataFrame, types=Dict(:site => String))
select!(df_soil_params, [:site, :Ksat, :theta_s])

# --- G. 基础静态因素 ---
df_static = CSV.read(path_static, DataFrame, types=Dict(:site => String))
select!(df_static, Not([:lon, :lat, :n])) 

# 🚀 世纪大拼接
df_merged = innerjoin(df_target_daily, df_dyn_daily, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_sm_daily, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_rain_features, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_vpd_lai, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_static, on=:site)
df_merged = innerjoin(df_merged, df_twi, on=:site)                  
df_merged = innerjoin(df_merged, df_soil_params, on=:site)          

# 💡 特征工程合成
df_merged.Month_Sin = sin.(2 * π * month.(df_merged.date_only) / 12)
df_merged.Month_Cos = cos.(2 * π * month.(df_merged.date_only) / 12)
df_merged.Max_Hourly_Intensity = df_merged.ERA5_Peak_Ratio .* df_merged.Prcp_Current_day

# 🎯 目标与特征定义
target_col = "Inf_daily_Sum" 
final_features =[
    "Max_Hourly_Intensity", "VPD", "LAI",               
    "Month_Sin", "Month_Cos", "Temp_C_initial",         
    "Soil_Water_L1_initial", "Soil_Water_L2_initial", 
    "Ksat", "theta_s",                                  
    "Sand_0cm_percent", "Clay_0cm_percent", "BulkDensity_0cm_g_cm3", "OrganicCarbon_0cm_g_kg", 
    "Elevation_m", "Slope_degrees", "TWI",              
    "Prcp_Current_day", "Prcp_Lag1_day", "Prcp_Lag2_day", 
    "Prcp_Past_3d_sum", "Prcp_Past_7d_sum"            
]

all_cols = unique([["site", "date_only", target_col]; final_features])
select!(df_merged, all_cols)

# 📌 更新：明确跳过土壤含水率存在缺失值的行
println("🧹 正在清理缺失数据 (特别检查一、二层土壤含水率)...")
dropmissing!(df_merged, ["Soil_Water_L1_initial", "Soil_Water_L2_initial"])
# 再全局清理一下其他变量偶然出现的 missing，保证矩阵能完全输入到模型中
dropmissing!(df_merged)

filter!(row -> row.Prcp_Current_day > 0.1 || row.Inf_daily_Sum > 0.0, df_merged)

println("✅ 数据清洗及拼接完成！无缺失值的有效高质量样本行数: $(nrow(df_merged))")


# ====================================================================
# 2. ML 相关性分析模块 (VIF + 随机森林重要度)
# ====================================================================
function calculate_vif(df_features::DataFrame)
    cols = names(df_features)
    vif_values = fill(1.0, length(cols))
    
    is_numeric =[eltype(df_features[!, c]) <: Number for c in cols]
    numeric_cols = cols[is_numeric]
    
    if length(numeric_cols) > 1
        X_mat = Matrix{Float64}(df_features[:, numeric_cols])
        
        std_vals = std(X_mat, dims=1)
        for j in 1:length(std_vals)
            if std_vals[j] == 0.0
                X_mat[:, j] .+= randn(size(X_mat, 1)) * 1e-6 
            end
        end
        
        R = cor(X_mat)
        R_ridge = R + I(size(R, 1)) * 1e-5 
        inv_R = inv(R_ridge)
        vif_numeric = diag(inv_R)
        
        num_idx = 1
        for (i, is_num) in enumerate(is_numeric)
            if is_num
                vif_values[i] = vif_numeric[num_idx]
                num_idx += 1
            end
        end
    end
    return vif_values
end

function stepwise_vif_selection(df_X::DataFrame, threshold=5.0)
    df_curr = copy(df_X)
    println("\n--- [Phase 2] VIF Collinearity Check (Threshold: $threshold) ---")
    while true
        vifs = calculate_vif(df_curr)
        max_vif, idx = findmax(vifs)
        cols = names(df_curr)
        if max_vif < threshold
            println("✅ VIF Check Passed!")
            break
        end
        if length(cols) <= 1 break end
            removed_col = cols[idx]
        println("❌ Removing Variable:[ $removed_col ] (VIF = $(round(max_vif, digits=2)))")
        select!(df_curr, Not(removed_col))
    end
    return df_curr, names(df_curr)
end

function analyze_infiltration_factors(df::DataFrame, target_col::String, out_dir::String)
    println("\n====== [2/2] Starting Feature Analysis ======")
    
    feature_names_all = filter(x -> x != target_col, names(df[!, Not(["site", "date_only"])]))
    
    # ⚠️ 终极清洗：彻底剥离 Union{Missing}，并将所有潜在的 NaN 替换为 0.0
    X_all = DataFrame()
    for col in feature_names_all
        X_all[!, col] = replace(Vector{Float64}(df[!, col]), NaN => 0.0)
    end
    y = replace(Vector{Float64}(df[!, target_col]), NaN => 0.0)
    
    # ---------------------------------------------------------
    # 🔄 [Phase 1 调整为先做 VIF 剔除] 
    # ---------------------------------------------------------
    X_clean_df, kept_features = stepwise_vif_selection(X_all, 5.0) 
    
    # ---------------------------------------------------------
    # 🔄 [Phase 2 斯皮尔曼相关性及热力图] 仅使用 VIF 通过的特征
    # ---------------------------------------------------------
    println("\n---[Phase 2] Spearman Correlation (VIF Cleaned) ---")
    spearman_scores = Dict{String, Float64}()
    
    # 虽然热力图只画保留的，但我们仍需要算出所有特征的 rho 用于最终结果表格对比
    for col in feature_names_all
        if std(X_all[!, col]) == 0.0 || std(y) == 0.0
            spearman_scores[col] = 0.0
        else
            rho = corspearman(X_all[!, col], y)
            spearman_scores[col] = isnan(rho) ? 0.0 : rho
        end
    end
    
    # 🎨 核心修改：画图数据只放入 kept_features 和 target_col
    all_cols_for_plot = [kept_features; target_col]
    data_matrix = Matrix(hcat(X_clean_df, DataFrame(target=y)))
    cor_matrix = corspearman(data_matrix)
    replace!(cor_matrix, NaN => 0.0)
    
    # 稍微缩小一点画板，因为变量变少了，同时调大字体让图表更清晰
    p1 = heatmap(cor_matrix, c=:RdBu_10, clims=(-1,1), 
                 title="Spearman Correlation (VIF Cleaned)",
                 xticks=(1:length(all_cols_for_plot), all_cols_for_plot), 
                 yticks=(1:length(all_cols_for_plot), all_cols_for_plot), 
                 xrotation=60, yflip=true, size=(1000, 900), annot=true,
                 xtickfontsize=10, ytickfontsize=10, annotfontsize=8,
                 left_margin=25Plots.mm, bottom_margin=25Plots.mm)
                 
    display(p1)
    savefig(p1, joinpath(out_dir, "1_FullPhysics_Heatmap_VIF_Cleaned.png"))

    # ---------------------------------------------------------
    # [Phase 3 随机森林重要度]
    # ---------------------------------------------------------
    println("\n---[Phase 3] Random Forest Importance ---")
    X_matrix = Matrix{Float64}(X_clean_df)
    y_vec = Vector{Float64}(y)
    
    rf_model = build_forest(y_vec, X_matrix, -1, 100, 0.7, 6)
    
    base_mse = mean((apply_forest(rf_model, X_matrix) .- y_vec).^2)
    rf_importances = Dict{String, Float64}()
    
    for (i, name) in enumerate(kept_features)
        X_perm = copy(X_matrix)
        X_perm[:, i] = shuffle(X_perm[:, i])
        perm_mse = mean((apply_forest(rf_model, X_perm) .- y_vec).^2)
        rf_importances[name] = max(0.0, perm_mse - base_mse)
    end
    
    total_imp = sum(values(rf_importances))
    total_imp = total_imp == 0 ? 1.0 : total_imp
    for k in keys(rf_importances)
        rf_importances[k] /= total_imp
    end

    # ---------------------------------------------------------
    # 汇总结果表
    # ---------------------------------------------------------
    results = DataFrame(Factor = String[], VIF_Check = String[], Linear_Rho = Float64[], 
                        RF_Imp_Pct = Float64[], Conclusion = String[])
    
    NONLINEAR_THRES = 0.025
    
    for name in feature_names_all
        is_kept = name in kept_features
        rho = spearman_scores[name]
        rf_imp = is_kept ? rf_importances[name] : 0.0
        
        decision = !is_kept ? "❌ Redundant" : (rf_imp > NONLINEAR_THRES ? "🌟 Key Driver" : "🗑️ Noise")
        push!(results, (name, is_kept ? "Pass" : "Drop", round(rho, digits=3), round(rf_imp, digits=3), decision))
    end
    
    sort!(results, :RF_Imp_Pct, rev=true)
    display(results)
    CSV.write(joinpath(out_dir, "Feature_Selection_Summary.csv"), results)

    # ---------------------------------------------------------
    # 图 2. 决策散点图
    # ---------------------------------------------------------
    if !isempty(kept_features)
        plot_df = filter(row -> row.VIF_Check == "Pass", results)
        
        key_df = filter(r -> occursin("Key Driver", r.Conclusion), plot_df)
        noise_df = filter(r -> !occursin("Key Driver", r.Conclusion), plot_df)
        
        max_linear = maximum(abs.(plot_df.Linear_Rho))
        max_rf = maximum(plot_df.RF_Imp_Pct)
        x_limit = max_linear > 0 ? max_linear * 1.35 : 0.15
        y_limit = max_rf > 0 ? max_rf * 1.15 : 0.8
        
        p2 = plot(
            xlabel = "Linear Strength (|Spearman Rho|)",
            ylabel = "Non-linear Importance (Random Forest)",
            title = "Factor Selection Matrix (Target: Absolute Inf)",
            size = (1000, 750),
            xlims = (0.0, max(0.15, x_limit)),
            ylims = (0.0, max(1, y_limit)), 
            grid = true, gridalpha = 0.3,
            legend = :topright,
            left_margin = 8Plots.mm, bottom_margin = 8Plots.mm
        )
        
        hline!([NONLINEAR_THRES], color=:blue, linestyle=:dash, linewidth=1.5, label="Non-linear Thres ($NONLINEAR_THRES)")
        
        if nrow(noise_df) > 0
            scatter!(abs.(noise_df.Linear_Rho), noise_df.RF_Imp_Pct, 
                label = "Weak Factors", color = :slategray, markersize = 8, markerstrokewidth = 0, alpha = 0.6)
        end

        if nrow(key_df) > 0
            scatter!(abs.(key_df.Linear_Rho), key_df.RF_Imp_Pct, 
                label = "Key Drivers", color = :crimson, markersize = 12, markerstrokecolor = :black, markerstrokewidth = 1.5)
            
            for row in eachrow(key_df)
                annotate!(abs(row.Linear_Rho), row.RF_Imp_Pct + (y_limit * 0.035), text(row.Factor, 10, "Helvetica", :black, :center))
            end
        end
        
        display(p2)
        save_path2 = joinpath(out_dir, "2_FullPhysics_Decision_Matrix_Absolute.png")
        savefig(p2, save_path2)
        println("✅ 决策矩阵图已保存至: $save_path2")
    end
    
    return results
end

# ====================================================================
# 3. 执行分析
# ====================================================================
results = analyze_infiltration_factors(df_merged, target_col, output_dir)
println("\n🎉 分析完成！")
