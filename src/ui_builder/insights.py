import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def business_insights_view(data):
    """Display the Business Insights & Recommendations page"""

    st.title("Business Insights & Recommendations")

    if data.empty:
        st.warning("No sales data available.")
        return

    st.markdown(
        "This page analyzes your historical sales data and provides **actionable "
        "recommendations** to help boost revenue."
    )

    # Executive Summary
    display_executive_summary(data)

    st.divider()

    # Top & Bottom Performers
    display_performer_analysis(data)

    st.divider()

    # Seasonal & Timing Insights
    display_timing_insights(data)

    st.divider()

    # Growth Opportunities
    display_growth_opportunities(data)

    st.divider()

    # Actionable Recommendations
    display_recommendations(data)


def display_executive_summary(data):
    """Show a high-level executive summary of sales performance"""

    st.header("Executive Summary")

    total_sales = data["sales"].sum()
    total_days = (data["date"].max() - data["date"].min()).days + 1
    avg_daily_revenue = data.groupby("date")["sales"].sum().mean()

    num_stores = data["store_id"].nunique() if "store_id" in data.columns else 0
    num_items = data["item_id"].nunique() if "item_id" in data.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales (all time)", f"${total_sales:,.0f}")
    with col2:
        st.metric("Avg Daily Revenue", f"${avg_daily_revenue:,.0f}")
    with col3:
        st.metric("Active Stores", f"{num_stores}")
    with col4:
        st.metric("Product Range", f"{num_items} items")

    # Month-over-month trend
    monthly = (
        data.groupby(data["date"].dt.to_period("M"))["sales"]
        .sum()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["pct_change"] = monthly["sales"].pct_change() * 100

    recent_trend = monthly["pct_change"].tail(3).mean()
    if recent_trend > 5:
        st.success(
            f"**Positive trend:** Sales have been growing ~{recent_trend:.1f}% "
            "month-over-month in recent months. Keep investing in what's working."
        )
    elif recent_trend > 0:
        st.info(
            f"**Stable growth:** Sales grew ~{recent_trend:.1f}% month-over-month "
            "recently. There's room to accelerate with targeted strategies."
        )
    else:
        st.warning(
            f"**Declining trend:** Sales dropped ~{abs(recent_trend):.1f}% "
            "month-over-month recently. Review the recommendations below to reverse this."
        )

    # Monthly revenue chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["date"], monthly["sales"], color="steelblue", width=20)
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Total Sales ($)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_performer_analysis(data):
    """Analyze top and bottom performing stores and products"""

    st.header("Performance Analysis")

    col1, col2 = st.columns(2)

    # --- Store Performance ---
    with col1:
        st.subheader("Store Performance Ranking")

        store_col = "store_name" if "store_name" in data.columns else "store_id"
        store_sales = (
            data.groupby(store_col)["sales"]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        store_sales.columns = [store_col, "Total Sales", "Avg Sale", "Transactions"]

        best_store = store_sales.iloc[0]
        worst_store = store_sales.iloc[-1]

        st.success(
            f"**Top store:** {best_store[store_col]} "
            f"(${best_store['Total Sales']:,.0f} total, "
            f"${best_store['Avg Sale']:,.1f} avg per transaction)"
        )
        st.error(
            f"**Lowest store:** {worst_store[store_col]} "
            f"(${worst_store['Total Sales']:,.0f} total, "
            f"${worst_store['Avg Sale']:,.1f} avg per transaction)"
        )

        gap_pct = (
            (best_store["Total Sales"] - worst_store["Total Sales"])
            / worst_store["Total Sales"]
            * 100
        )
        st.info(
            f"The gap between top and bottom stores is **{gap_pct:.0f}%**. "
            "Investigate what the top store does differently — staffing, "
            "layout, local marketing — and replicate those practices."
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2ecc71" if i < 3 else "#e74c3c" if i >= len(store_sales) - 3 else "#3498db"
                  for i in range(len(store_sales))]
        ax.barh(store_sales[store_col], store_sales["Total Sales"], color=colors)
        ax.set_xlabel("Total Sales ($)")
        ax.set_title("Store Revenue Ranking")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Product Performance ---
    with col2:
        st.subheader("Product Performance Ranking")

        item_col = "item_name" if "item_name" in data.columns else "item_id"
        item_sales = (
            data.groupby(item_col)["sales"]
            .agg(["sum", "mean"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        item_sales.columns = [item_col, "Total Sales", "Avg Daily Sales"]

        top5 = item_sales.head(5)
        bottom5 = item_sales.tail(5)

        st.success(
            f"**Best sellers:** {', '.join(top5[item_col].tolist())} — "
            "ensure these are always well-stocked and prominently displayed."
        )
        st.error(
            f"**Lowest sellers:** {', '.join(bottom5[item_col].tolist())} — "
            "consider promotions, bundling with popular items, or reducing shelf space."
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        top10 = item_sales.head(10)
        ax.barh(top10[item_col], top10["Total Sales"], color="#2ecc71")
        ax.set_xlabel("Total Sales ($)")
        ax.set_title("Top 10 Products by Revenue")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Category Performance ---
    if "category" in data.columns:
        st.subheader("Category Performance")

        cat_sales = (
            data.groupby("category")["sales"]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
            .reset_index()
        )
        cat_sales.columns = ["Category", "Total Sales", "Avg Sale", "Transactions"]
        cat_sales["Revenue Share (%)"] = (
            cat_sales["Total Sales"] / cat_sales["Total Sales"].sum() * 100
        ).round(1)

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(
                cat_sales["Total Sales"],
                labels=cat_sales["Category"],
                autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("Set2", len(cat_sales)),
            )
            ax.set_title("Revenue Share by Category")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            top_cat = cat_sales.iloc[0]
            bottom_cat = cat_sales.iloc[-1]
            st.markdown(
                f"**{top_cat['Category']}** leads with **{top_cat['Revenue Share (%)']:.1f}%** "
                f"of total revenue (${top_cat['Total Sales']:,.0f}).\n\n"
                f"**{bottom_cat['Category']}** contributes only "
                f"**{bottom_cat['Revenue Share (%)']:.1f}%** "
                f"(${bottom_cat['Total Sales']:,.0f}).\n\n"
                "**Recommendation:** Cross-sell low-performing categories alongside "
                "top performers. For example, bundle them in promotions or place them "
                "near high-traffic product areas."
            )


def display_timing_insights(data):
    """Analyze when sales peak and provide scheduling recommendations"""

    st.header("Timing & Seasonal Insights")

    col1, col2 = st.columns(2)

    # --- Day of Week ---
    with col1:
        st.subheader("Day-of-Week Pattern")

        data_copy = data.copy()
        data_copy["day_of_week"] = data_copy["date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        dow_sales = (
            data_copy.groupby("day_of_week")["sales"]
            .mean()
            .reindex(day_order)
        )

        best_day = dow_sales.idxmax()
        worst_day = dow_sales.idxmin()
        weekend_avg = dow_sales[["Saturday", "Sunday"]].mean()
        weekday_avg = dow_sales[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean()
        weekend_lift = (weekend_avg - weekday_avg) / weekday_avg * 100

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#e74c3c" if d in ["Saturday", "Sunday"] else "#3498db" for d in day_order]
        ax.bar(day_order, dow_sales.values, color=colors)
        ax.set_ylabel("Avg Sales ($)")
        ax.set_title("Average Sales by Day of Week")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if weekend_lift > 0:
            st.info(
                f"**Weekends drive {weekend_lift:.1f}% more sales** than weekdays. "
                f"**{best_day}** is the best day; **{worst_day}** is the weakest.\n\n"
                "**Action:** Schedule promotions and extra staffing on weekends. "
                f"Consider mid-week flash sales on **{worst_day}** to boost traffic."
            )
        else:
            st.info(
                f"Weekdays outperform weekends. **{best_day}** is the best day.\n\n"
                "**Action:** Focus on weekday foot traffic and consider weekend specials."
            )

    # --- Monthly Seasonality ---
    with col2:
        st.subheader("Monthly Seasonality")

        monthly_avg = data.groupby(data["date"].dt.month)["sales"].mean()
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        available_months = [month_names[m - 1] for m in monthly_avg.index]

        peak_month = month_names[monthly_avg.idxmax() - 1]
        low_month = month_names[monthly_avg.idxmin() - 1]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(available_months, monthly_avg.values, "o-", color="steelblue", linewidth=2)
        ax.fill_between(available_months, monthly_avg.values, alpha=0.15, color="steelblue")
        ax.set_ylabel("Avg Sales ($)")
        ax.set_title("Monthly Sales Seasonality")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"**Peak season:** {peak_month} — ensure sufficient inventory and staff.\n\n"
            f"**Low season:** {low_month} — run clearance sales or loyalty promotions to "
            "maintain customer engagement.\n\n"
            "**Action:** Plan inventory purchasing 1-2 months ahead of peak season. "
            "Use the low season for store improvements and staff training."
        )


def display_growth_opportunities(data):
    """Identify specific growth opportunities from the data"""

    st.header("Growth Opportunities")

    # --- Store x Category cross-analysis ---
    if "category" in data.columns and "store_name" in data.columns:
        st.subheader("Underperforming Store-Category Combinations")

        store_cat = (
            data.groupby(["store_name", "category"])["sales"]
            .mean()
            .reset_index()
        )
        pivot = store_cat.pivot(index="store_name", columns="category", values="sales")

        # Find the biggest gaps
        category_avg = pivot.mean()
        gaps = pivot.copy()
        for col in gaps.columns:
            gaps[col] = ((pivot[col] - category_avg[col]) / category_avg[col] * 100).round(1)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            gaps,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": "% vs Category Average"},
        )
        ax.set_title("Store Performance vs Category Average (%)")
        ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            "**How to read this chart:** Green cells indicate above-average performance; "
            "red cells are below average. Focus improvement efforts on the deep-red "
            "combinations — these represent the biggest revenue recovery opportunities."
        )

        # Find top 3 opportunities
        opportunities = []
        for store in gaps.index:
            for cat in gaps.columns:
                val = gaps.loc[store, cat]
                if val < -10:
                    opportunities.append((store, cat, val))

        opportunities.sort(key=lambda x: x[2])

        if opportunities:
            st.subheader("Top Improvement Targets")
            for store, cat, gap in opportunities[:5]:
                potential = abs(gap) * category_avg[cat] / 100
                st.markdown(
                    f"- **{store}** / **{cat}**: {gap:.0f}% below average. "
                    f"Closing this gap could add ~${potential:,.0f} per day."
                )

    # --- Year-over-year growth by item ---
    if data["date"].dt.year.nunique() > 1:
        st.subheader("Year-over-Year Item Growth")

        item_col = "item_name" if "item_name" in data.columns else "item_id"
        yoy = data.groupby([data["date"].dt.year, item_col])["sales"].sum().unstack(level=0)

        if len(yoy.columns) >= 2:
            year1, year2 = sorted(yoy.columns)[:2]
            yoy["growth_pct"] = ((yoy[year2] - yoy[year1]) / yoy[year1] * 100).round(1)
            yoy = yoy.sort_values("growth_pct", ascending=False).dropna()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Fastest Growing Products**")
                top_growth = yoy.head(5)
                for item in top_growth.index:
                    pct = top_growth.loc[item, "growth_pct"]
                    st.markdown(f"- **{item}**: +{pct:.1f}% growth")
                st.markdown(
                    "*Invest more in these products — increase stock and "
                    "expand their display area.*"
                )

            with col2:
                st.markdown("**Declining Products**")
                declining = yoy[yoy["growth_pct"] < 0].tail(5)
                if not declining.empty:
                    for item in declining.index:
                        pct = declining.loc[item, "growth_pct"]
                        st.markdown(f"- **{item}**: {pct:.1f}%")
                    st.markdown(
                        "*Review pricing, placement, and relevance. Consider "
                        "promotions or replacing with trending alternatives.*"
                    )
                else:
                    st.success("No declining products found!")


def display_recommendations(data):
    """Generate and display actionable business recommendations"""

    st.header("Actionable Recommendations")

    st.markdown(
        "Based on the analysis above, here are the **top strategies** to boost revenue:"
    )

    # Calculate key metrics for recommendations
    store_col = "store_name" if "store_name" in data.columns else "store_id"
    store_sales = data.groupby(store_col)["sales"].sum()
    best_store = store_sales.idxmax()
    worst_store = store_sales.idxmin()

    data_copy = data.copy()
    data_copy["day_of_week"] = data_copy["date"].dt.day_name()
    dow_sales = data_copy.groupby("day_of_week")["sales"].mean()
    worst_day = dow_sales.idxmin()

    monthly_avg = data.groupby(data["date"].dt.month)["sales"].mean()
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    peak_month = month_names[monthly_avg.idxmax() - 1]
    low_month = month_names[monthly_avg.idxmin() - 1]

    item_col = "item_name" if "item_name" in data.columns else "item_id"
    top_items = data.groupby(item_col)["sales"].sum().nlargest(5).index.tolist()
    bottom_items = data.groupby(item_col)["sales"].sum().nsmallest(5).index.tolist()

    recommendations = [
        {
            "title": "Replicate Top Store Practices",
            "detail": (
                f"**{best_store}** significantly outperforms **{worst_store}**. "
                "Conduct a store audit: compare staffing levels, product placement, "
                "local marketing, and customer service quality. Transfer proven "
                "practices from top to bottom performers."
            ),
            "impact": "High",
        },
        {
            "title": "Optimize Weekly Promotions",
            "detail": (
                f"**{worst_day}** has the lowest sales. Launch targeted mid-week "
                f"promotions (e.g., '{worst_day} Deals') with 10-15% discounts on "
                "slow-moving items to drive foot traffic on weak days."
            ),
            "impact": "Medium",
        },
        {
            "title": "Seasonal Inventory Planning",
            "detail": (
                f"Sales peak in **{peak_month}** and dip in **{low_month}**. "
                f"Stock up 1-2 months before {peak_month}. During {low_month}, "
                "run loyalty programs and clearance sales to maintain cash flow."
            ),
            "impact": "High",
        },
        {
            "title": "Product Bundling Strategy",
            "detail": (
                f"Bundle low sellers ({', '.join(bottom_items[:3])}) with top sellers "
                f"({', '.join(top_items[:3])}). Offer 'Buy X, get Y at 20% off' "
                "to move slow inventory while boosting basket size."
            ),
            "impact": "Medium",
        },
        {
            "title": "Focus on High-Value Products",
            "detail": (
                f"Top revenue drivers are: **{', '.join(top_items)}**. "
                "Ensure these are always in stock, prominently displayed, "
                "and featured in marketing materials."
            ),
            "impact": "High",
        },
    ]

    if "category" in data.columns:
        cat_sales = data.groupby("category")["sales"].sum()
        top_cat = cat_sales.idxmax()
        bottom_cat = cat_sales.idxmin()
        recommendations.append(
            {
                "title": "Cross-Category Promotion",
                "detail": (
                    f"**{bottom_cat}** underperforms compared to **{top_cat}**. "
                    f"Place {bottom_cat} products near {top_cat} displays. "
                    "Run cross-category promotions to increase exposure."
                ),
                "impact": "Medium",
            }
        )

    for i, rec in enumerate(recommendations, 1):
        impact_color = {"High": "red", "Medium": "orange", "Low": "blue"}[rec["impact"]]
        st.markdown(
            f"### {i}. {rec['title']}  \n"
            f"**Impact:** :{impact_color}[{rec['impact']}]\n\n"
            f"{rec['detail']}"
        )
        st.divider()
