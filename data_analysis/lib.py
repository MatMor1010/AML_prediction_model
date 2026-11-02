import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from plotly.subplots import make_subplots


def eda(df, target, feature, bins=10):
    """Plot the count (bars) and the mean target (line) for each group of a feature."""
    y = df[target]
    x = df[feature]

    # Text or categorical target: use the share of its last level
    if not is_numeric_dtype(y):
        last_level = sorted(y.dropna().unique())[-1]
        y = y.eq(last_level).astype(float).where(y.notna())

    # Numeric feature: group many values into ranges, otherwise just shorten the labels
    if is_numeric_dtype(x) and x.nunique() > bins:
        edges = x.drop_duplicates().quantile(np.linspace(0, 1, bins + 1))
        x = pd.cut(x, edges, include_lowest=True)
    elif is_numeric_dtype(x):
        x = x.round(3)

    # Count and mean target for each group
    summary = y.groupby(x, observed=True, dropna=False).agg(["size", "mean"]).reset_index()
    groups = summary[feature].astype(str).fillna("missing")

    # Bars on the left axis, line on the right axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=groups, y=summary["size"], name="Count", marker_color="red", opacity=0.3))
    fig.add_trace(go.Scatter(x=groups, y=summary["mean"], name=target, mode="lines+markers",
                             line_color="blue"), secondary_y=True)

    # Titles and style
    fig.update_layout(title=f"Mean {target} by {feature}", template="plotly_white",
                      font=dict(size=16, color="black"))
    fig.update_xaxes(title_text=feature, type="category")
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text=f"Mean {target}", secondary_y=True, showgrid=False, tickmode="auto")

    return fig
