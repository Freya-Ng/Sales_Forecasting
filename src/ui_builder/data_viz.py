import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sales_time_series(filtered_data, selected_store, selected_store_name):
    """Plot sales over time, aggregated daily."""
    daily_sales = filtered_data.groupby("date")["sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_sales["date"], daily_sales["sales"], color="steelblue", linewidth=1)
    ax.fill_between(daily_sales["date"], daily_sales["sales"], alpha=0.15, color="steelblue")

    store_label = (
        selected_store_name
        if selected_store_name != "All Stores"
        else selected_store
    )
    title = (
        f"Daily Sales - {store_label}"
        if store_label != "All Stores"
        else "Daily Sales - All Stores"
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_day_of_week_pattern(filtered_data):
    """Plot average sales by day of the week."""
    data = filtered_data.copy()
    data["day_of_week"] = data["date"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_avg = (
        data.groupby("day_of_week")["sales"]
        .mean()
        .reindex(day_order)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=daily_avg, x="day_of_week", y="sales", ax=ax, color="steelblue")
    ax.set_title("Average Sales by Day of Week")
    ax.set_xlabel("")
    ax.set_ylabel("Average Sales")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_sales_distribution(filtered_data):
    """Plot histogram of sales values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_data["sales"], bins=50, kde=True, ax=ax, color="steelblue")
    ax.set_title("Sales Distribution")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_category_distribution(filtered_data):
    """Plot pie chart of sales by category."""
    category_sales = filtered_data.groupby("category")["sales"].sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        category_sales,
        labels=category_sales.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title("Sales by Category")
    fig.tight_layout()
    return fig


def plot_store_comparison(filtered_data, store_identifier):
    """Plot horizontal bar chart comparing top 10 stores by total sales."""
    store_sales = (
        filtered_data.groupby(store_identifier)["sales"]
        .sum()
        .sort_values(ascending=True)
        .tail(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    store_sales.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 10 Stores by Sales")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig
