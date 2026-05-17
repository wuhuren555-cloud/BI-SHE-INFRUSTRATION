using CSV
using DataFrames
using Plots
using Dates
using Printf
using Statistics
using Random

# ==============================================================================
# 1. 配置路径与变量名
# ==============================================================================
BASE_DIR = "C:\\Users\\26332\\OneDrive\\Desktop\\kong mission\\mechin learning\\code\\infiltration"

SIM_CSV   = joinpath(BASE_DIR, "SoilDiffEqs.jl-master", "结果图", "Campbell-3hourly-NSE(重筛选4-10月-全图像)", "All_Sites_SimData_Merged.csv")
PARAM_CSV = joinpath(BASE_DIR, "SoilDiffEqs.jl-master", "结果图", "Campbell-3hourly-NSE(重筛选4-10月-全图像)", "All_Sites_Params_Merged.csv")
RAIN_CSV  = joinpath(BASE_DIR, "data", "rainfall", "prcp_CHM_PRE_V2_ChinaSM_2018-2019_sp2702.csv")

# 🌟 输出文件：强调是 3h 尺度数据，但带有日尺度约束
OUT_CSV   = joinpath(BASE_DIR, "SoilDiffEqs.jl-master", "结果图", "Campbell-3hourly-NSE(重筛选4-10月-全图像)", "3Hourly_Infiltration_Calculated_DailyScaled.csv")
# 🌟 绘图文件夹：强调是日尺度聚合图
IMG_OUTPUT_DIR = joinpath(BASE_DIR, "下渗时序图", "包含负值", "日尺度绘图_输出3h数据_随机100站")

mkpath(dirname(OUT_CSV))
mkpath(IMG_OUTPUT_DIR)

ENV["GKSwstype"] = "nul"

# ==============================================================================
# 2. 核心算法与时间解析函数
# ==============================================================================

function estimate_real_theta_r(theta_s, b)
    ratio = clamp(0.05 + 0.02 * b, 0.05, 0.45)
    return theta_s * ratio
end

function parse_to_datetime(d_str)
    s = strip(string(d_str))
    s = replace(s, "/" => "-")
    s = replace(s, " " => "T")
    if count(==(':'), s) == 1 s = s * ":00" end
    s = length(s) >= 19 ? s[1:19] : s
    try return DateTime(s) catch 
        return missing end
end

function calc_Q_gravity(theta_bottom, p)
    S = theta_bottom / p.theta_s
    S = clamp(S, 0.0, 1.0)
    Ksat_mm_h = p.Ksat * 10.0 
    Q_rate = Ksat_mm_h * (S ^ (2.0 * p.b + 3.0))
    return Q_rate 
end

safe_sum(x) = isempty(collect(skipmissing(x))) ? 0.0 : sum(skipmissing(x))

# ==============================================================================
# 3. 主程序：计算与绘图
# ==============================================================================

function main()
    println("="^60)
    println("🚀 开始计算: 导出 3h 尺度下渗特征，绘图聚合至日尺度")
    println("="^60)

    println("📥 读取 CSV (SimData & Params)...")
    df_sim = CSV.read(SIM_CSV, DataFrame; silencewarnings=true)
    df_param = CSV.read(PARAM_CSV, DataFrame; silencewarnings=true)
    dropmissing!(df_param) 
    
    df_sim.site = string.(strip.(string.(df_sim.site)))
    df_param.site = string.(strip.(string.(df_param.site)))

    println("📥 读取降雨数据并按 【日尺度 (Daily)】 聚合...")
    df_rain = CSV.read(RAIN_CSV, DataFrame)
    df_rain.Site_ID = string.(strip.(string.(df_rain.site)))
    df_rain.Date = Date.(parse_to_datetime.(df_rain.time)) 
    dropmissing!(df_rain, :Date)
    
    # 按 Date 分组计算日总降雨
    gdf_rain_daily = combine(groupby(df_rain,[:Site_ID, :Date]), :prec => safe_sum => :Rain_Daily_Sum)
    dict_rain = groupby(gdf_rain_daily, :Site_ID)

    sim_sites = unique(df_sim.site)
    param_sites = unique(df_param.site)
    rain_sites = unique(df_rain.Site_ID)
    
    sites = intersect(sim_sites, param_sites, rain_sites)
    
    Random.seed!(42) 
    num_to_plot = min(100, length(sites))
    sites_to_plot = Set(shuffle(sites)[1:num_to_plot])
    
    println("🎲 共有 $(length(sites)) 个站点参与计算。\n")

    CURRENT_DELTA_Z =[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0]
    THETA_COLS =["Depth_10", "Depth_20", "Depth_30", "Depth_40", "Depth_50", "Depth_60", "Depth_80", "Depth_100"]

    all_3h_inf = DataFrame()
    gr() 
    default(fontfamily="Helvetica", guidefontsize=10, tickfontsize=8)

    for (i, site_id) in enumerate(sites)
        
        p_site = filter(:site => ==(site_id), df_param)
        if nrow(p_site) != length(THETA_COLS) continue end
        sort!(p_site, :depth) 

        p_bottom = p_site[end, :] 
        real_theta_r_arr = [estimate_real_theta_r(row.theta_s, row.b) for row in eachrow(p_site)]
        
        sub_sim = filter(:site => ==(site_id), df_sim)
        sub_sim.DateTime = parse_to_datetime.(sub_sim.time)
        dropmissing!(sub_sim, :DateTime)
        dropmissing!(sub_sim, THETA_COLS)
        sort!(sub_sim, :DateTime)
        
        if nrow(sub_sim) < 2 continue end
        
        rain_dict_daily = Dict{Date, Float64}()
        if haskey(dict_rain, (Site_ID = site_id,))
            for row in eachrow(dict_rain[(Site_ID = site_id,)])
                rain_dict_daily[row.Date] = row.Rain_Daily_Sum
            end
        end
        
        n_steps = nrow(sub_sim)
        I_vol_3h_raw = zeros(Float64, n_steps - 1)
        I_vol_3h_final = zeros(Float64, n_steps - 1) # 最终受比例约束的 3h 下渗
        
        theta_matrix = Matrix{Float64}(sub_sim[!, THETA_COLS])
        S_array = [sum(theta_matrix[r, :] .* CURRENT_DELTA_Z) for r in 1:n_steps]
        
        # ==========================================================
        # 步骤 1: 计算 3 小时尺度基础潜力
        # ==========================================================
        for t in 2:n_steps
            theta_bottom = theta_matrix[t-1, end] 
            Q_rate_mm_h = calc_Q_gravity(theta_bottom, p_bottom)
            Q_3h_theoretical = Q_rate_mm_h * 3.0
            
            drainable_water_3h = 0.0
            for idx in 1:length(THETA_COLS)
                drainable_water_3h += max(0.0, theta_matrix[t-1, idx] - real_theta_r_arr[idx]) * CURRENT_DELTA_Z[idx]
            end
            
            Q_3h_actual = min(Q_3h_theoretical, drainable_water_3h)
            delta_S = S_array[t] - S_array[t-1]
            
            I_calc = delta_S + Q_3h_actual 
            I_vol_3h_raw[t-1] = max(0.0, I_calc) 
        end
        
        # ==========================================================
        # 步骤 2: 降雨日尺度校验与 3h 比例缩放
        # ==========================================================
        dates_sim = Date.(sub_sim.DateTime[1:end-1]) 
        unique_dates = unique(dates_sim)
        
        for d in unique_dates
            idx_for_day = findall(==(d), dates_sim)
            P_daily = get(rain_dict_daily, d, 0.0)
            I_daily_raw_sum = sum(I_vol_3h_raw[idx_for_day])
            
            # 按比例扣减，将结果赋值给 I_vol_3h_final
            if I_daily_raw_sum > P_daily
                if I_daily_raw_sum > 0.0
                    ratio = P_daily / I_daily_raw_sum
                    I_vol_3h_final[idx_for_day] = I_vol_3h_raw[idx_for_day] .* ratio
                else
                    I_vol_3h_final[idx_for_day] .= 0.0
                end
            else
                I_vol_3h_final[idx_for_day] = I_vol_3h_raw[idx_for_day]
            end
        end
        
        # 🌟 核心修改 1：CSV 输出保持 3 小时高分辨率
        inf_df = DataFrame(
            Site_ID = site_id, 
            DateTime = sub_sim.DateTime[1:end-1], 
            Inf_3h_Final = I_vol_3h_final
        )
        append!(all_3h_inf, inf_df)
        
        # ==========================================================
        # 🌟 核心修改 2：绘图前对 3h 结果进行纯正的日尺度聚合
        # ==========================================================
        if site_id in sites_to_plot
            I_daily_for_plot = Float64[]
            P_daily_for_plot = Float64[]
            
            # 将该站的 3h 下渗和降雨重新加总到每一天
            for d in unique_dates
                idx_for_day = findall(==(d), dates_sim)
                push!(I_daily_for_plot, sum(I_vol_3h_final[idx_for_day]))
                push!(P_daily_for_plot, get(rain_dict_daily, d, 0.0))
            end
            
            l = @layout[a{0.3h}; b]
            
            p_rain = plot(unique_dates, P_daily_for_plot, seriestype = :bar, yflip = true, color = :blue, linecolor = :blue, 
                          label = "Rainfall (Daily)", ylabel = "P (mm/d)", xformatter = _ -> "", 
                          bottom_margin = -2Plots.mm, grid = false, frame = :box)
            max_r = maximum(P_daily_for_plot)
            if max_r > 0 plot!(p_rain, ylims=(0, max_r * 1.2)) end
            
            p_inf = plot(unique_dates, I_daily_for_plot, seriestype = :bar, color = :orange, linecolor = :orange, 
                         label = "Infiltration (Daily)", ylabel = "Inf (mm/d)", xlabel = "Date", 
                         top_margin = -2Plots.mm, xrotation = 45, legend = :topright, frame = :box)
            max_i = isempty(I_daily_for_plot) ? 0.0 : maximum(I_daily_for_plot)
            plot!(p_inf, ylims=(0, max_i <= 0 ? 1.0 : max_i * 1.2))
            
            final_plot = plot(p_rain, p_inf, layout = l, size = (800, 600), dpi = 300, link = :x, 
                              title =["Site $site_id: Daily Hyetograph & Infiltration" ""])
            
            outfile = joinpath(IMG_OUTPUT_DIR, "Site_$(site_id).png")
            savefig(final_plot, outfile)
            
            print("\r进度 [$i / $(length(sites))] Site $site_id 计算并绘图完成        ")
        else
            print("\r进度 [$i / $(length(sites))] Site $site_id 计算完成            ")
        end
    end
    
    if nrow(all_3h_inf) > 0
        CSV.write(OUT_CSV, all_3h_inf)
        println("\n\n🎉 任务完成！")
        println("📊 数据已保存: 3 小时分辨率的特征数据 (用于 XGBoost) -> $OUT_CSV")
        println("🖼️ 图表已保存: 日尺度聚合的直观质量校验图 -> $IMG_OUTPUT_DIR")
    else
        println("\n⚠️ 未生成任何数据。")
    end
end

main()
