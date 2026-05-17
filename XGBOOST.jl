using DataFrames
using CSV
using Dates
using Statistics
using StatsBase
using Plots
using StatsPlots
using Random
using XGBoost
using Printf
using NCDatasets  

# ====================================================================
# 0. 基础配置与路径设置 
# ====================================================================
Random.seed!(42) 
default(fontfamily="sans-serif") 

# --- 基础数据路径 ---
path_daily_rain   = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\rainfall\prcp_CHM_PRE_V2_ChinaSM_2018-2019_sp2702.csv"
path_dynamic      = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Sites_Dynamic_Hourly_2018_2019.csv"
path_static       = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Sites_Static_Factors.csv"  
path_infiltration = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\SoilDiffEqs.jl-master\结果图\Campbell-3hourly-NSE(重筛选4-10月-全图像)\3Hourly_Infiltration_Calculated_DailyScaled.csv" 
path_vpd          = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\XG _DATA\Dynamic_VPD_Daily.csv"
path_obs_sm       = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\data\sm\CHinaSM_hourly_2018-2019.nc"

# 输出文件夹更新名称
output_dir        = raw"C:\Users\26332\OneDrive\Desktop\kong mission\mechin learning\code\infiltration\XGBoost_13Features_调参/"
if !isdir(output_dir) mkpath(output_dir) end

# 时间解析辅助函数
parse_dt(x::DateTime) = x; parse_dt(x::Date) = DateTime(x)
parse_dt(x::AbstractString) = tryparse(DateTime, x) === nothing ? DateTime(x, "yyyy-mm-dd HH:MM:SS") : DateTime(x)
parse_dt(::Missing) = missing
parse_date(x::Date) = x; parse_date(x::DateTime) = Date(x)
parse_date(x::AbstractString) = tryparse(Date, x) === nothing ? Date(x, "yyyy-mm-dd") : Date(x)
parse_date(::Missing) = missing
safe_datetime(dt) = DateTime(Dates.year(dt), Dates.month(dt), Dates.day(dt), Dates.hour(dt), Dates.minute(dt), Dates.second(dt))

# ====================================================================
# 1. 读取并拼接 13 个核心特征
# ====================================================================
println("📌[1/4] 正在加载并融合 13 个核心物理与时间特征...")

df_rain = CSV.read(path_daily_rain, DataFrame, types=Dict(:site => String))
df_rain.time = parse_date.(df_rain.time)
sort!(df_rain, [:site, :time])
df_rain_features = combine(groupby(df_rain, :site)) do group
    df_out = DataFrame(date_only = group.time)
    df_out.Prcp_Current_day = group.prec  
    past_7d = Vector{Union{Missing, Float64}}(missing, nrow(group))
    for i in 8:nrow(group) past_7d[i] = sum(skipmissing(group.prec[i-7:i-1])) end
    df_out.Prcp_Past_7d_sum = past_7d
    return df_out 
end

df_dyn = CSV.read(path_dynamic, DataFrame, types=Dict(:site => String))
df_dyn.datetime = parse_dt.(df_dyn.datetime)
df_dyn.date_only = Date.(df_dyn.datetime)
df_dyn_daily = combine(groupby(df_dyn, [:site, :date_only])) do group
    era5_hourly_rain = coalesce.(group.Precipitation_mm, 0.0)
    era5_daily_sum = sum(era5_hourly_rain)
    era5_max_hourly = maximum(era5_hourly_rain)
    peak_ratio = era5_daily_sum > 0.0 ? (era5_max_hourly / era5_daily_sum) : 0.0
    return DataFrame(ERA5_Peak_Ratio = peak_ratio)
end

ds = NCDataset(path_obs_sm, "r")
nc_sites = string.(ds["site"][:]); nc_times = safe_datetime.(ds["time"][:])
sm_var = ds["SM"]; dim_names = dimnames(sm_var); depth_dim_idx = findfirst(==("depth"), dim_names)
nc_sm_L1 = collect(selectdim(sm_var, depth_dim_idx, 1)); nc_sm_L2 = collect(selectdim(sm_var, depth_dim_idx, 2)) 
dim_names_2d = filter(!=("depth"), dim_names)
sites_vec = String[]; times_vec = DateTime[]; sm_L1_vec = Union{Missing, Float64}[]; sm_L2_vec = Union{Missing, Float64}[]

if dim_names_2d[1] == "site" || dim_names_2d[1] == "station"
    for i in 1:length(nc_sites), j in 1:length(nc_times)
        push!(sites_vec, nc_sites[i]); push!(times_vec, nc_times[j]); push!(sm_L1_vec, nc_sm_L1[i, j]); push!(sm_L2_vec, nc_sm_L2[i, j])
    end
else
    for j in 1:length(nc_times), i in 1:length(nc_sites)
        push!(sites_vec, nc_sites[i]); push!(times_vec, nc_times[j]); push!(sm_L1_vec, nc_sm_L1[j, i]); push!(sm_L2_vec, nc_sm_L2[j, i])
    end
end
close(ds)

df_obs_sm = DataFrame(site=sites_vec, datetime=times_vec, Soil_Water_L1_initial=sm_L1_vec, Soil_Water_L2_initial=sm_L2_vec)
df_obs_sm.date_only = Date.(df_obs_sm.datetime)
df_sm_daily = combine(groupby(df_obs_sm, [:site, :date_only])) do group
    vals_L1 = collect(skipmissing(group.Soil_Water_L1_initial))
    vals_L2 = collect(skipmissing(group.Soil_Water_L2_initial))
    return DataFrame(Soil_Water_L1_initial = isempty(vals_L1) ? missing : first(vals_L1), Soil_Water_L2_initial = isempty(vals_L2) ? missing : first(vals_L2))
end
df_sm_daily.Soil_Water_L1_initial = df_sm_daily.Soil_Water_L1_initial ./ 100.0
df_sm_daily.Soil_Water_L2_initial = df_sm_daily.Soil_Water_L2_initial ./ 100.0

df_target = CSV.read(path_infiltration, DataFrame, types=Dict(:Site_ID => String), missingstring=["na", "NA", "NaN", ""])
rename!(df_target, :Site_ID => :site, :DateTime => :time_3h)
dropmissing!(df_target, :Inf_3h_Final)
df_target.Inf_3h_Final = Float64.(df_target.Inf_3h_Final)
filter!(row -> isfinite(row.Inf_3h_Final), df_target) 

df_target.time_3h = parse_dt.(df_target.time_3h)
df_target.date_only = Date.(df_target.time_3h)
df_target_daily = combine(groupby(df_target, [:site, :date_only]), :Inf_3h_Final => sum => :Inf_daily_Sum)

df_vpd = CSV.read(path_vpd, DataFrame, types=Dict(:site => String))
df_vpd.date_only = parse_date.(df_vpd.date_only)
rename!(df_vpd, :mean => :VPD); select!(df_vpd, [:site, :date_only, :VPD])

df_static = CSV.read(path_static, DataFrame, types=Dict(:site => String))
select!(df_static, [:site, :Elevation_m, :Sand_0cm_percent, :Clay_0cm_percent, :BulkDensity_0cm_g_cm3, :OrganicCarbon_0cm_g_kg])

df_merged = innerjoin(df_target_daily, df_dyn_daily, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_sm_daily, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_rain_features, on=[:site, :date_only])
df_merged = innerjoin(df_merged, df_vpd, on=[:site, :date_only])  
df_merged = innerjoin(df_merged, df_static, on=:site)

# 💡 时间特征双全 (Sin 和 Cos 构成完整的圆)
df_merged.Month_Cos = cos.(2 * π * month.(df_merged.date_only) / 12)
df_merged.Month_Sin = sin.(2 * π * month.(df_merged.date_only) / 12)
df_merged.Max_Hourly_Intensity = df_merged.ERA5_Peak_Ratio .* df_merged.Prcp_Current_day

target_col = "Inf_daily_Sum" 
final_features = ["Prcp_Current_day", "Max_Hourly_Intensity", "Soil_Water_L1_initial", "VPD", 
                  "Month_Cos", "Month_Sin", "Soil_Water_L2_initial", "Elevation_m", 
                  "BulkDensity_0cm_g_cm3", "Sand_0cm_percent", "OrganicCarbon_0cm_g_kg", 
                  "Clay_0cm_percent", "Prcp_Past_7d_sum"]

all_cols = unique([["site", "date_only", target_col]; final_features])
select!(df_merged, all_cols)
dropmissing!(df_merged)  
filter!(row -> isfinite(row.Inf_daily_Sum), df_merged) 

filter!(row -> row.Inf_daily_Sum <= row.Prcp_Current_day + 50.0, df_merged)
filter!(row -> row.Prcp_Current_day > 0.1 || row.Inf_daily_Sum > 0.0, df_merged)

println("✅ 完美拼接完成！13特征配置，有效高质量样本行数: $(nrow(df_merged))")

# ====================================================================
# 2. 空间折划分及 NSE 计算函数
# ====================================================================
println("\n📌 [2/4] 正在按站点执行 5-Fold 空间划分...")
unique_sites = unique(df_merged.site); shuffle!(unique_sites) 
K_FOLDS = 5; fold_size = ceil(Int, length(unique_sites) / K_FOLDS)
site_folds = Dict{String, Int}()
for (i, site) in enumerate(unique_sites) site_folds[site] = min(K_FOLDS, ceil(Int, i / fold_size)) end
df_merged.Fold = [site_folds[s] for s in df_merged.site]

function calc_nse(obs, pred)
    var_obs = sum((obs .- mean(obs)).^2)
    if var_obs == 0 || length(obs) < 5 return NaN end
    return 1 - (sum((obs .- pred).^2) / var_obs)
end

# ====================================================================
# 3. XGBoost 自动网格搜索调参 (串行记录版，恢复纯粹的损失函数)
# ====================================================================
println("\n🔍 [3/4] 🚀 开始执行网格搜索 (Grid Search) 串行模式...")

# 生成参数网格 (3x3x3 = 27 种组合)
param_combinations = []
for md in [4, 6, 8], sub in [0.7, 0.8, 0.9], col in [0.7, 0.8, 0.9]
    push!(param_combinations, (max_depth=md, subsample=sub, colsample_bytree=col))
end

best_nse = -Inf
best_params = param_combinations[1]
best_predictions = zeros(Float32, nrow(df_merged))
best_imp_scores = zeros(Float64, length(final_features))

df_grid_results = DataFrame(Iteration = Int[], Max_Depth = Int[], Subsample = Float64[], Colsample_Bytree = Float64[], Global_NSE = Float64[])

for (i, p) in enumerate(param_combinations)
    @printf("▶️ 正在训练组合 [%2d/%d]: max_depth=%d | subsample=%.1f | colsample=%.1f ... ", 
            i, length(param_combinations), p.max_depth, p.subsample, p.colsample_bytree)
    
    temp_preds = zeros(Float32, nrow(df_merged))
    temp_imp = zeros(Float64, length(final_features))
    
    for k in 1:K_FOLDS
        test_idx  = df_merged.Fold .== k; train_candidate_idx = df_merged.Fold .!= k
        candidate_sites = unique(df_merged[train_candidate_idx, :site]); shuffle!(candidate_sites)
        val_split_point = floor(Int, 0.8 * length(candidate_sites))
        
        actual_train_sites = candidate_sites[1:val_split_point]
        actual_train_idx = train_candidate_idx .& in.(df_merged.site, Ref(actual_train_sites))
        val_idx          = train_candidate_idx .& .!actual_train_idx
        
        X_train = Matrix{Float32}(df_merged[actual_train_idx, final_features])
        y_train = Vector{Float32}(df_merged[actual_train_idx, target_col]) 
        X_val   = Matrix{Float32}(df_merged[val_idx, final_features])
        y_val   = Vector{Float32}(df_merged[val_idx, target_col])
        X_test  = Matrix{Float32}(df_merged[test_idx, final_features])
        
        # 🌟 彻底去除了 weight 权重，恢复最纯粹的数据规律拟合
        dtrain = DMatrix(X_train, y_train)
        dval   = DMatrix(X_val, y_val)
        dtest  = DMatrix(X_test) 
        
        bst = xgboost(dtrain, 
            num_round = 800,               
            max_depth = p.max_depth,               
            eta = 0.02,                    
            subsample = p.subsample,               
            colsample_bytree = p.colsample_bytree, 
            objective = "reg:squarederror", 
            eval_metric = "rmse",
            watchlist = (train = dtrain, eval = dval), 
            early_stopping_rounds = 50,
            print_every_n = 1000 # 静默输出避免刷屏
        )
        
        pred_amount = XGBoost.predict(bst, dtest)
        
        # 🌟 保留 max(0)，防止负数，但不设置上限钳制
        temp_preds[test_idx] .= max.(pred_amount, 0.0)
        
        imp_raw = XGBoost.importance(bst)
        for (idx, val_array) in imp_raw
            if 1 <= idx <= length(final_features) temp_imp[idx] += val_array[1] end
        end
    end
    
    current_nse = calc_nse(df_merged.Inf_daily_Sum, temp_preds)
    @printf("✅ 得分: %.4f\n", current_nse)
    
    push!(df_grid_results, (i, p.max_depth, p.subsample, p.colsample_bytree, current_nse))
    
    if current_nse > best_nse
        global best_nse = current_nse
        global best_params = p
        best_predictions .= temp_preds
        best_imp_scores .= temp_imp
    end
end

println("\n=====================================================")
println("🏆 网格搜索完成！找到最优参数组合：")
println("   Max Depth        : ", best_params.max_depth)
println("   Subsample        : ", best_params.subsample)
println("   Colsample_bytree : ", best_params.colsample_bytree)
println("   最佳全局 NSE 得分 : ", round(best_nse, digits=4))
println("=====================================================\n")

df_merged.Predicted_Absolute = best_predictions
global_imp_scores = best_imp_scores

sort!(df_grid_results, :Global_NSE, rev=true) 
CSV.write(joinpath(output_dir, "0_GridSearch_All_Trials_Summary.csv"), df_grid_results)
println("💾 网格搜索统计结果已保存至：0_GridSearch_All_Trials_Summary.csv")

# ====================================================================
# 4. 全局 CV 评估与出图 
# ====================================================================
println("\n📊 [4/4] 开始出图与结果评估...")

idx_inf_only = df_merged.Inf_daily_Sum .> 0.0
cv_nse_inf_only = calc_nse(df_merged.Inf_daily_Sum[idx_inf_only], df_merged.Predicted_Absolute[idx_inf_only])
println("=> 🎯 最优 5-Fold CV 局部 NSE (Obs>0): $(round(cv_nse_inf_only, digits=4))")

max_v = maximum([maximum(df_merged.Inf_daily_Sum), maximum(df_merged.Predicted_Absolute)])
p1 = scatter(df_merged.Inf_daily_Sum, df_merged.Predicted_Absolute, label="OOB CV Samples", color=:steelblue, alpha=0.2,
    xlabel="Observed Daily Infiltration (mm)", ylabel="Predicted Daily Infiltration (mm)", 
    title=@sprintf("Final XGBoost Model (13 Feat. Pure)\nGlobal NSE=%.3f | Local NSE=%.3f", best_nse, cv_nse_inf_only), 
    legend=:topleft, aspect_ratio=:equal, titlefontsize=11)
plot!(p1, [0, max_v], [0, max_v], color=:red, lw=2, linestyle=:dash, label="1:1 Line")
savefig(p1, joinpath(output_dir, "1_CV_Daily_Scatter.png"))

global_imp_scores ./= sum(global_imp_scores) 
sort_idx = sortperm(global_imp_scores)
max_imp = maximum(global_imp_scores)  
p2 = bar(1:length(final_features), global_imp_scores[sort_idx], orientation = :h,
    yticks = (1:length(final_features), final_features[sort_idx]),
    xlabel = "Average CV Importance (XGBoost Gain)", legend = false, color = :teal, 
    title = "Feature Importance (Final Tuned Model)", margin = 15Plots.mm,
    xlims = (0, max_imp * 1.15), size = (800, 600))
savefig(p2, joinpath(output_dir, "2_CV_Feature_Importance.png"))


# ====================================================================
# 5. 抽取站点绘制时序图
# ====================================================================
site_plot_dir = joinpath(output_dir, "Random_Sites_Plots")
if !isdir(site_plot_dir) mkpath(site_plot_dir) end
test_sites = unique(df_merged.site)
selected_sites = sample(test_sites, min(500, length(test_sites)), replace=false)

for site in selected_sites
    df_site = filter(row -> row.site == site, df_merged); sort!(df_site, :date_only)
    site_nse = calc_nse(df_site.Inf_daily_Sum, df_site.Predicted_Absolute)
    nse_str = isnan(site_nse) ? "N/A" : "$(round(site_nse, digits=3))"
    max_rain = maximum(df_site.Prcp_Current_day); max_rain = max_rain <= 0.0 ? 10.0 : max_rain
    max_inf = maximum([maximum(df_site.Inf_daily_Sum), maximum(df_site.Predicted_Absolute)]); max_inf = max_inf <= 0.0 ? 5.0 : max_inf

    p_rain = plot(df_site.date_only, df_site.Prcp_Current_day, seriestype = :sticks, color = :blue, linewidth = 2.5, label = "Rainfall",
        yflip = true, ylims = (0, max_rain * 1.2), ylabel = "P (mm/d)", title = "Site: $site  |  Tuned NSE: $nse_str", xticks = :none, legend = :topright, bottom_margin = 0Plots.mm)
    p_inf = plot(df_site.date_only, df_site.Inf_daily_Sum, label = "Observed", color = :steelblue, linewidth = 2,
        xlabel = "Date", ylabel = "Inf (mm/d)", ylims = (0, max_inf * 1.3), legend = :topright, top_margin = 0Plots.mm)
    plot!(p_inf, df_site.date_only, df_site.Predicted_Absolute, label = "Predicted", color = :darkorange, linewidth = 2, linestyle = :dash)
    p_combined = plot(p_rain, p_inf, layout = grid(2, 1, heights=[0.3, 0.7]), size = (1000, 600), link = :x, margin = 5Plots.mm)
    savefig(p_combined, joinpath(site_plot_dir, "Site_$(site)_Hyetograph.png"))
end


# ====================================================================
# 6. 站点级 NSE 统计与区间划分 
# ====================================================================
all_unique_sites = unique(df_merged.site)
site_nses = Float64[]; valid_sites = String[]

for site in all_unique_sites
    df_site = filter(row -> row.site == site, df_merged)
    site_nse = calc_nse(df_site.Inf_daily_Sum, df_site.Predicted_Absolute)
    if !isnan(site_nse) push!(site_nses, site_nse); push!(valid_sites, site) end
end

count_great = sum(site_nses .>= 0.6); count_good  = sum(0.5 .<= site_nses .< 0.6)
count_fair  = sum(0.0 .<= site_nses .< 0.5); count_poor  = sum(site_nses .< 0.0); total_valid = length(site_nses)

println("\n🎯 最终优化的站点级预测 NSE 统计结果 (有效评估站点数: $total_valid):")
println(@sprintf("  [1] 优秀 (NSE >= 0.6):       %4d 站  (%.2f %%)", count_great, count_great / total_valid * 100))
println(@sprintf("  [2] 良好 (0.5 <= NSE < 0.6): %4d 站  (%.2f %%)", count_good,  count_good / total_valid * 100))
println(@sprintf("  [3] 一般 (0.0 <= NSE < 0.5): %4d 站  (%.2f %%)", count_fair,  count_fair / total_valid * 100))
println(@sprintf("  [4] 较差 (NSE < 0.0):        %4d 站  (%.2f %%)", count_poor,  count_poor / total_valid * 100))

df_site_summary = DataFrame(Site = valid_sites, NSE = site_nses)
sort!(df_site_summary, :NSE, rev=true); CSV.write(joinpath(output_dir, "3_All_Sites_NSE_Summary.csv"), df_site_summary)

cat_labels = ["NSE >= 0.6", "0.5 <= NSE < 0.6", "0.0 <= NSE < 0.5", "NSE < 0.0"]; cat_counts = [count_great, count_good, count_fair, count_poor]; colors = [:forestgreen, :dodgerblue, :orange, :crimson]
p_bar_nse = bar(cat_labels, cat_counts, title = "Distribution of Site NSE (Tuned)", ylabel = "Number of Sites", legend = false, color = colors, size = (750, 500), margin = 6Plots.mm)
for (i, count) in enumerate(cat_counts) annotate!(p_bar_nse, i, count + (maximum(cat_counts)*0.04), text("$(count) sites", 10, :bottom)) end
savefig(p_bar_nse, joinpath(output_dir, "4_Site_NSE_Bar.png"))
savefig(pie(cat_labels, cat_counts, title = "Percentage of Site NSE (Tuned)", color = colors, legend = :outertopright, size = (700, 500)), joinpath(output_dir, "5_Site_NSE_Pie.png"))


# ====================================================================
# 7. 保存预测结果到 CSV 文件 (还原为连续日历)
# ====================================================================
df_events = select(df_merged, [:site, :date_only, :Prcp_Current_day, :Inf_daily_Sum, :Predicted_Absolute])
rename!(df_events, :Inf_daily_Sum => :Observed_Infiltration, :Predicted_Absolute => :Predicted_Infiltration)

df_continuous_skeleton = select(df_rain, [:site, :time]); rename!(df_continuous_skeleton, :time => :date_only)
filter!(row -> row.site in valid_sites, df_continuous_skeleton) 

df_final_output = leftjoin(df_continuous_skeleton, df_events, on=[:site, :date_only])
df_final_output.Prcp_Current_day = coalesce.(df_final_output.Prcp_Current_day, 0.0)
df_final_output.Observed_Infiltration = coalesce.(df_final_output.Observed_Infiltration, 0.0)
df_final_output.Predicted_Infiltration = coalesce.(df_final_output.Predicted_Infiltration, 0.0)
sort!(df_final_output, [:site, :date_only])

CSV.write(joinpath(output_dir, "All_Sites_Daily_Infiltration_Predictions_Continuous.csv"), df_final_output)
println("✅ 连续时间序列数据导出完毕！文件存储于：$(output_dir)")
