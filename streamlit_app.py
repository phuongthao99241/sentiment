import streamlit as st
import pandas as pd
import datetime
from prophet import Prophet
import numpy as np
import os
import plotly.express as px

st.set_page_config(
    page_title="📊 TV-Stimmungsanalyse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("logo_transparent.png", width=200)
    st.markdown("## SentimentInsights")
    tab_choice = st.radio("Navigiere zu:", ["Dashboard", "KI-Planung", "Vergleich", "Bericht"])

st.title("📺 KI-gestützte TV-Stimmungsanalyse")


@st.cache_data
def generate_mock_comments(program, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    data = []
    rng = np.random.default_rng(seed=hash(program) % 2**32)

    # Basisstimmung je nach Programm
    if program == "Master Chef":
        base_probs = [0.7, 0.2, 0.1]
    elif program == "The Voice":
        base_probs = [0.55, 0.25, 0.2]
    else:
        base_probs = [0.5, 0.3, 0.2]

    for date in dates:
        weekday = date.weekday()
        n_comments = rng.integers(20, 40)

        # Wochenend-Effekt: mehr Kommentare freitags bis sonntags
        if weekday >= 4:
            n_comments += rng.integers(10, 20)

        # Schwankung der Stimmung (leicht sinusartig)
        mood_shift = 0.1 * np.sin(date.dayofyear / 10.0 + (0 if program == "Master Chef" else 1))

        # Sonderfall: negativer Peak (z. B. Skandal)
        if rng.random() < 0.1 and weekday == 2:  # Mittwoch = Drama
            shock = [-0.2, 0.0, 0.2]
        else:
            shock = [0.0, 0.0, 0.0]

        probs = np.array(base_probs) + np.array([mood_shift, -0.05, -mood_shift]) + shock
        probs = np.clip(probs, 0.05, 0.9)
        probs = probs / probs.sum()

        sentiments = rng.choice(["positive", "neutral", "negative"], size=n_comments, p=probs)
        for s in sentiments:
            data.append({
                "date": date,
                "program": program,
                "text": f"Kommentar zu {program} am {date.date()}",
                "sentiment": s
            })

    return pd.DataFrame(data)



def save_dataset(df, program):
    filename = f"sentiment_data_{program.replace(' ', '_')}.csv"
    if os.path.exists(filename):
        existing = pd.read_csv(filename, parse_dates=["date"])
        df = pd.concat([existing, df]).drop_duplicates(subset=["date", "text"])  # optional: Duplikate vermeiden
    df.to_csv(filename, index=False)




def simulate_forecast(df: pd.DataFrame, program: str, periods: int = 14):
    """Erstelle eine Prophet-Prognose und generiere eine umsetzbare Empfehlung."""
    df_prog = df[df["program"] == program]
    grouped = (
        df_prog.groupby("date").apply(lambda x: (x["sentiment"] == "positive").mean()).reset_index()
    )
    grouped.columns = ["ds", "y"]

    model = Prophet(daily_seasonality=True)
    model.fit(grouped)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Nur Zukunft ab heute anzeigen
    start_forecast = pd.Timestamp.today().normalize()
    forecast_future = forecast[forecast["ds"] >= start_forecast]

    # Plot
    fig = px.line(
        forecast_future,
        x="ds",
        y="yhat",
        title="Prognose: Positiver Stimmungsanteil (nächste 14 Tage)",
        markers=True,
        line_shape="spline",
    )
    fig.update_traces(line_color="#A8E6A3")
    fig.update_layout(yaxis_title="Anteil positiv", xaxis_title="Datum")

    # --- Empfehlung ---------------------------------------------------
    base = grouped["y"].mean()
    trend_14d = forecast_future["yhat"].mean()
    trend_3d = forecast_future["yhat"].tail(3).mean()
    delta_abs = trend_3d - base
    delta_pp = delta_abs * 100

    # Neuer Bestandteil: Steigung der Trendlinie
    slope = forecast_future["yhat"].iloc[-1] - forecast_future["yhat"].iloc[0]

    def fmt(p):
        return f"{p:.1%}"

    # Visuell-logisch korrektere Klassifizierung
    if slope > 0.01:
        sign = "steigt"
    elif slope < -0.01:
        sign = "sinkt"
    elif abs(delta_abs) < 0.01:
        sign = "bleibt stabil"
    else:
        sign = "sinkt" if delta_abs < 0 else "steigt"

    text_rcmd = str

    if sign == "steigt":
        text_rcmd = "📈 Stimmung steigt – nutze das Momentum für Promotion."
    elif sign == "sinkt":
        text_rcmd = "📉 Stimmung sinkt – Inhalte überprüfen und anpassen."
    else:
        text_rcmd = "📊 Stimmung stabil – keine Maßnahmen erforderlich."

    recommendation = (
        f"**Kurzbewertung**: Die Stimmung {sign} "
        f"(Basis: {fmt(base)}, Trend 3 Tage: {fmt(trend_3d)}, Δ ≈ {delta_pp:+.1f} pp).\n\n"
        f"{text_rcmd}"
    )

    return fig, forecast, grouped, recommendation


AVAILABLE_SHOWS = ["Master Chef", "The Voice"]
TODAY = datetime.date.today()

if tab_choice == "Dashboard":
    st.subheader("🎛️ Auswahl")
    col_sel1, col_sel2, col_sel3 = st.columns([2,3,1])

    with col_sel1:
        program = st.selectbox("Wähle ein Programm", AVAILABLE_SHOWS, key="dash_prog")
    with col_sel2:
        date_range = st.date_input(
            "Zeitraum wählen",
            [TODAY - datetime.timedelta(days=6), TODAY],
            max_value=TODAY,
            key="dash_daterange"
        )

    with col_sel3:
        start_btn = st.button("Analyse starten", key="dash_start")

    if start_btn:
        if len(date_range) != 2:
            st.error("Bitte wähle einen Start- und Endzeitpunkt aus.")
            st.stop()

        df = generate_mock_comments(program, date_range[0], date_range[1])
        df["weekday"] = df["date"].dt.day_name()
        save_dataset(df, program)

        # KPIs
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("📄 Kommentare", len(df))
        col_kpi2.metric("😊 Positive", (df["sentiment"] == "positive").sum())
        col_kpi3.metric("☹️ Negative", (df["sentiment"] == "negative").sum())

        st.divider()

        #col_main1, col_main2, col_main3 = st.columns([1.2, 2, 2])
        col_main1, col_main2 = st.columns([3,3])

        with col_main1:
            st.subheader("📊 Verteilung")
            sentiment_count = df["sentiment"].value_counts().reset_index()
            sentiment_count.columns = ["Sentiment", "Anzahl"]
            #fig_pie = px.pie(sentiment_count, names="Sentiment", values="Anzahl", hole=0.4)
            fig_pie = px.pie(
                sentiment_count,
                names="Sentiment",
                values="Anzahl",
                hole=0.4,
                color="Sentiment",  # wichtig für Zuordnung
                color_discrete_map={
                    "positive": "#A8E6A3",  # hellgrün
                    "neutral": "#ADD8E6",  # hellblau
                    "negative": "#F4A6A6"  # hellrot
                }
            )

            st.plotly_chart(fig_pie, use_container_width=True)

            #st.subheader("📝 Beispielkommentare")
            #st.dataframe(df[["date", "text", "sentiment"]].sample(5), use_container_width=True)

            st.subheader("🔍 Histogramm")
            fig_hist = px.histogram(
                df, x="sentiment", color="sentiment",
                color_discrete_map={
                    "positive": "#A8E6A3",
                    "neutral": "#ADD8E6",
                    "negative": "#F4A6A6"
                },
                title="Verteilung der Stimmung"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("📉 Anteil positiver Kommentare pro Tag")
            df_trend = df.groupby("date").apply(
                lambda x: (x["sentiment"] == "positive").sum() / len(x)
            ).reset_index(name="Positiver Anteil")
            fig_trend = px.line(
                df_trend,
                x="date",
                y="Positiver Anteil",
                markers=True,
                line_shape="spline",
                title="Verlauf des positiven Anteils"
            )
            fig_trend.update_traces(line_color="#A8E6A3")  # hellgrün
            st.plotly_chart(fig_trend, use_container_width=True)



        with col_main2:
            st.subheader("📈 Sentiment über Zeit")
            df_day = df.groupby(["date", "sentiment"]).size().reset_index(name="Anzahl")
            #fig_time = px.bar(df_day, x="date", y="Anzahl", color="sentiment", barmode="group")
            fig_time = px.bar(
                df_day, x="date", y="Anzahl", color="sentiment", barmode="group",
                color_discrete_map={
                    "positive": "#A8E6A3",
                    "neutral": "#ADD8E6",
                    "negative": "#F4A6A6"
                }
            )

            st.plotly_chart(fig_time, use_container_width=True)

            st.subheader("📅 Stimmung nach Wochentag")
            df_wday = df.groupby(["weekday", "sentiment"]).size().reset_index(name="Anzahl")
            #fig_wday = px.bar(df_wday, x="weekday", y="Anzahl", color="sentiment", barmode="group")
            fig_wday = px.bar(
                df_wday, x="weekday", y="Anzahl", color="sentiment", barmode="group",
                color_discrete_map={
                    "positive": "#A8E6A3",
                    "neutral": "#ADD8E6",
                    "negative": "#F4A6A6"
                }
            )

            st.plotly_chart(fig_wday, use_container_width=True)

            # Heatmap nach Tag & Uhrzeit (wenn Uhrzeiten vorhanden sind)
            # Heatmap nach Tag & Uhrzeit (Plotly-Version)
            st.subheader("🔥 Aktivitäts-Hotspots (nach Wochentag und Stunde)")
            if "hour" not in df.columns:
                df["hour"] = np.random.choice(range(8, 23), size=len(df))

            df_heat = df.groupby(["weekday", "hour"]).size().reset_index(name="Kommentare")
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df_heat["weekday"] = pd.Categorical(df_heat["weekday"], categories=weekday_order, ordered=True)
            df_heat = df_heat.sort_values(["weekday", "hour"])

            fig_heat = px.density_heatmap(
                df_heat,
                x="hour",
                y="weekday",
                z="Kommentare",
                color_continuous_scale="YlOrRd",
                title="Heatmap der Kommentaraktivität",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        #with col_main3:
            #st.subheader("🔍 Histogramm")
            #fig_hist = px.histogram(df, x="sentiment", color="sentiment", title="Verteilung der Stimmung")
            #st.plotly_chart(fig_hist, use_container_width=True)

            #st.subheader("📥 Daten herunterladen")
            #csv = df.to_csv(index=False).encode("utf-8")
            #st.download_button("Exportiere als CSV", csv, f"{program}_sentiment.csv", "text/csv")
            # Am Ende der Seite: CSV Export
        st.divider()
        st.subheader("📥 Daten herunterladen")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Exportiere als CSV", csv, f"{program}_sentiment.csv", "text/csv")



elif tab_choice == "KI-Planung":
    st.subheader("🤖 KI-gestützte Prognose der positiven Stimmungen")
    program = st.selectbox("Programm auswählen", AVAILABLE_SHOWS, key="plan_prog")
    filename = f"sentiment_data_{program.replace(' ', '_')}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=["date"])
        if st.button("Prognose für die nächsten 14 Tage anzeigen"):
            fig, forecast, grouped, recommendation = simulate_forecast(df, program, periods=14)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**{recommendation}**")
    else:
        st.warning("Bitte analysiere zuerst Daten im Dashboard.")

elif tab_choice == "Vergleich":

    st.subheader("📈 Vergleich zwischen zwei Programmen")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        program_a = st.selectbox("Programm A auswählen", AVAILABLE_SHOWS, key="comp_a")
    with col_c2:
        remaining = [s for s in AVAILABLE_SHOWS if s != program_a]
        program_b = st.selectbox("Programm B auswählen", remaining, key="comp_b")

    st.markdown("\n**Zeitraum für Vergleich wählen**")
    compare_date_range = st.date_input(
        "Zeitraum eingrenzen",
        [TODAY - datetime.timedelta(days=14), TODAY],
        max_value=TODAY,
        key="compare_daterange"
    )

    if st.button("Vergleich anzeigen"):
        if len(compare_date_range) != 2:
            st.warning("Bitte Start- und Enddatum für den Vergleich auswählen.")
            st.stop()

        start_date = pd.to_datetime(compare_date_range[0]).normalize()
        end_date = pd.to_datetime(compare_date_range[1]).normalize()

        df_all = pd.DataFrame()
        for prog in [program_a, program_b]:
            file = f"sentiment_data_{prog.replace(' ', '_')}.csv"
            if os.path.exists(file):
                temp = pd.read_csv(file, parse_dates=["date"])
                temp["date"] = pd.to_datetime(temp["date"]).dt.normalize()
                temp = temp[(temp["date"] >= start_date) & (temp["date"] <= end_date)]
                df_all = pd.concat([df_all, temp])
            else:
                st.warning(f"Keine Daten für {prog} gefunden. Bitte zunächst im Dashboard analysieren.")

        if df_all.empty:
            st.warning("Keine Vergleichsdaten im gewählten Zeitraum vorhanden.")
        else:
            df_all["date"] = pd.to_datetime(df_all["date"]).dt.date

            # Anteil positive
            df_grouped = df_all.groupby(["date", "program"])
            df_ratio = df_grouped.apply(lambda x: (x["sentiment"] == "positive").sum() / len(x)).reset_index(name="Anteil Positiv")
            df_ratio = df_ratio.sort_values("date")

            fig_ratio = px.line(
                df_ratio,
                x="date",
                y="Anteil Positiv",
                color="program",
                title="Anteil positiver Kommentare",
            )
            fig_ratio.update_layout(xaxis=dict(tickformat="%d.%m.%Y"))
            fig_ratio.update_traces(line_shape="spline")
            st.plotly_chart(fig_ratio, use_container_width=True)

            # Anteil negativ
            df_ratio_neg = df_grouped.apply(lambda x: (x["sentiment"] == "negative").sum() / len(x)).reset_index(name="Anteil Negativ")
            df_ratio_neg = df_ratio_neg.sort_values("date")

            fig_ratio_neg = px.line(
                df_ratio_neg,
                x="date",
                y="Anteil Negativ",
                color="program",
                title="Anteil negativer Kommentare",
            )
            fig_ratio_neg.update_layout(xaxis=dict(tickformat="%d.%m.%Y"))
            fig_ratio_neg.update_traces(line_shape="spline")
            st.plotly_chart(fig_ratio_neg, use_container_width=True)


elif tab_choice == "Bericht":
    st.subheader("📄 Wochenbericht")
    df = pd.DataFrame()
    for prog in AVAILABLE_SHOWS:
        filename = f"sentiment_data_{prog.replace(' ', '_')}.csv"
        if os.path.exists(filename):
            df_prog = pd.read_csv(filename, parse_dates=["date"])
            df = pd.concat([df, df_prog])
    if not df.empty:
        df["KW"] = df["date"].dt.isocalendar().week
        report_programs = st.multiselect("Programme auswählen", AVAILABLE_SHOWS, default=AVAILABLE_SHOWS)
        df = df[df["program"].isin(report_programs)]
        report = df.groupby(["program", "KW", "sentiment"]).size().unstack(fill_value=0).reset_index()
        st.dataframe(report)
        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Bericht herunterladen", csv, "sentiment_report.csv", "text/csv")
    else:
        st.info("Bitte zuerst Daten analysieren.")

