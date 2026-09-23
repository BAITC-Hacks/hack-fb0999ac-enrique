"""Streamlit demo UI for the Beeline tariff campaign agent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

st.set_page_config(page_title="Enrique-agent | Beeline", page_icon="📡", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .hero {padding: 1.35rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#101b36,#182a4e); color: white; margin-bottom: 1rem;}
    .hero p {color: #d1ddf3; margin-bottom: 0;}
    .note {padding: .9rem 1rem; border-radius: 10px; background: #fff7e6; border: 1px solid #f5d38a; color: #553d12;}
    div[data-testid="stMetric"] {background:#f6f8fc; border:1px solid #e7ebf2; padding:14px; border-radius:12px;}
</style>
<div class="hero"><h1>Enrique-agent</h1><p>План кампаний по смене тарифа • Beeline Tariff Marketing Campaigns</p></div>
""", unsafe_allow_html=True)

st.markdown("""<div class="note"><b>Важно:</b> экран запускает выданного агента на локальной мок-среде.
Скорер и лимиты совпадают с механикой кейса, но эффекты в мок-модели отличаются от скрытых эффектов судейства.
Показанные суммы — результат локальной симуляции, а не прогноз финального балла.</div>""", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Демо")
    run_count = st.selectbox("Количество прогонов", [1, 10], index=0,
                             help="Один прогон использует seed 42; десять — seed 0–9 для проверки разброса.")
    run_button = st.button("Запустить агента", type="primary", use_container_width=True)
    st.divider()
    st.caption("Локальная проверка · без доступа к модели эффектов судейства")
    st.code("cd archive && python local_eval.py", language="bash")


def run_one(seed: int) -> dict:
    """Запускает Agent.act и тот же открытый скорер, что использует local_eval."""
    if not ARCHIVE.is_dir():
        raise FileNotFoundError("Не найдена папка archive рядом с app.py")
    previous_dir = Path.cwd()
    os.chdir(ARCHIVE)
    try:
        from agent import Agent
        from mock_environment import make_mock_env, _mock_fallback, _mock_impact_model
        from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns

        env, internals = make_mock_env(seed=seed)
        final_plan = Agent().act(env)
        final_plan = sanitize_campaigns(final_plan, env.tariffs)[:MAX_CAMPAIGNS]
        pilots = internals.executed_pilot_campaigns()
        all_rows = pd.DataFrame(pilots + final_plan)
        if all_rows.empty:
            return {"seed": seed, "error": "Агент не провёл пилотов и не вернул кампании."}
        filter_columns = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff"]
        for column in filter_columns + ["explicit_ids"]:
            if column not in all_rows.columns:
                all_rows[column] = None

        profile = env.customer_profile
        history = pd.read_csv("data/change_tariff.csv")
        model = _mock_impact_model(history)
        result = score_campaigns(
            all_rows, profile, model, env.tariffs, profile["predicted_arpu"].sum(),
            _mock_fallback, team_id="Enrique-agent-local",
        )
        result["n_pilots"] = len(pilots)
        result["seed"] = seed
        result["pilot_rows"] = pd.DataFrame(env.pilot_history)
        result["plan_rows"] = pd.DataFrame(final_plan)
        return result
    finally:
        os.chdir(previous_dir)


if run_button:
    seeds = [42] if run_count == 1 else list(range(10))
    outcomes = []
    with st.spinner("Агент проводит пилоты и собирает план кампаний…"):
        for seed in seeds:
            try:
                outcomes.append(run_one(seed))
            except Exception as exc:
                outcomes.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    st.session_state["demo_outcomes"] = outcomes

outcomes = st.session_state.get("demo_outcomes", [])
if not outcomes:
    st.subheader("Как проходит оценка")
    st.markdown("1. Агент видит профиль аудитории, тарифы, каналы и остаток лимитов.\n2. Проверяет часть гипотез шумными пилотами; их стоимость и контакты входят в общий лимит.\n3. Возвращает до 10 кампаний. Скорер учитывает кампании в порядке списка и не считает клиента дважды.")
    st.info("Нажми «Запустить агента», чтобы показать реальный локальный прогон и его ограничения.")
else:
    errors = [row for row in outcomes if row.get("error")]
    valid = [row for row in outcomes if not row.get("error")]
    for row in errors:
        st.error(f"Seed {row['seed']}: {row['error']}")

    if valid:
        focus = valid[0]
        st.subheader("Итог локального прогона" if run_count == 1 else "Итог и устойчивость")
        cols = st.columns(5)
        cols[0].metric("Чистый результат", f"{focus['net_arpu_gain']:,.0f} у.е.")
        cols[1].metric("Прирост до затрат", f"{focus['gross_arpu_lift']:,.0f} у.е.")
        cols[2].metric("Затраты связи", f"{focus['total_cost']:,.0f} у.е.")
        cols[3].metric("Пилоты / кампании", f"{focus['n_pilots']} / {focus['n_campaigns']}")
        cols[4].metric("Контакты", f"{focus['total_contacts']:,} / 15 000")

        if len(valid) > 1:
            nets = pd.Series([row["net_arpu_gain"] for row in valid], dtype=float)
            left, middle, right = st.columns(3)
            left.metric("Прогонов в плюс", f"{int((nets > 0).sum())} / {len(nets)}")
            middle.metric("Медианный net", f"{nets.median():,.0f} у.е.")
            right.metric("Диапазон net", f"{nets.min():,.0f} … {nets.max():,.0f}")
            st.caption("Разброс отражает разные шумные пилоты в мок-среде, не неопределённость скрытой модели судейства.")
            st.dataframe(pd.DataFrame([{"seed": r["seed"], "net, у.е.": r["net_arpu_gain"],
                                        "пилотов": r["n_pilots"], "кампаний": r["n_campaigns"]}
                                       for r in valid]), use_container_width=True, hide_index=True)

        st.progress(min(focus["budget_used_pct"] / 100, 1.0), text=f"Использовано бюджета: {focus['total_cost']:,.0f} / 100 000 у.е.")
        st.progress(min(focus["total_contacts"] / 15_000, 1.0), text=f"Использовано контактов: {focus['total_contacts']:,} / 15 000")

        tab_plan, tab_pilots, tab_explain = st.tabs(["Финальный план", "Результаты пилотов", "Формула и ограничения"])
        with tab_plan:
            plan = focus.get("plan_rows", pd.DataFrame())
            if plan.empty:
                st.warning("Агент не вернул итоговых кампаний.")
            else:
                display = plan.copy()
                display.insert(0, "№", range(1, len(display) + 1))
                rename = {"campaign_name": "Кампания", "filter_current_tariff": "Текущий тариф",
                          "filter_arpu_segment": "Сегмент ARPU", "filter_data_segment": "Интернет",
                          "filter_call_segment": "Звонки", "target_tariff": "Целевой тариф", "channel": "Канал"}
                display = display.rename(columns=rename)
                visible = [c for c in ["№", "Кампания", "Текущий тариф", "Сегмент ARPU", "Интернет", "Звонки", "Целевой тариф", "Канал"] if c in display]
                st.dataframe(display[visible], use_container_width=True, hide_index=True)
            details = pd.DataFrame(focus["campaigns_detail"])
            if not details.empty:
                st.caption("Показатели скорера по кампаниям (gross lift до дедупликации между кампаниями).")
                st.dataframe(details.rename(columns={"name": "Кампания", "channel": "Канал", "n_contacts": "Контакты",
                                                     "cost": "Расход, у.е.", "gross_lift": "Gross lift, у.е."}),
                             use_container_width=True, hide_index=True)
        with tab_pilots:
            pilots = focus.get("pilot_rows", pd.DataFrame())
            if pilots.empty:
                st.info("Пилоты не запускались.")
            else:
                display = pilots.rename(columns={"pilot": "Пилот", "target_tariff": "Целевой тариф", "channel": "Канал",
                                                 "n_customers": "Клиенты", "cost": "Стоимость, у.е.",
                                                 "observed_lift_ratio": "Наблюдаемый эффект (шумный)",
                                                 "observed_lift_total": "Наблюдаемый прирост (шумный)",
                                                 "remaining_budget": "Остаток бюджета", "remaining_contacts": "Остаток контактов"})
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.caption("Пилот даёт зашумлённое наблюдение. Это сигнал для выбора кампаний, а не точная оценка будущего эффекта.")
        with tab_explain:
            st.markdown("**Чистый результат = прирост ARPU (gross) − стоимость контактов.** Пилоты входят в общий счёт, клиенты дедуплицируются, лимиты применяются скорером.")
            st.markdown("**Ограничения кейса:** до 20 пилотов; 10–200 клиентов в пилоте; до 10 финальных кампаний; максимум 5 000 клиентов в кампании; 15 000 контактов и 100 000 у.е. общего бюджета.")
            st.warning("В локальной среде подставлена открытая мок-модель. Скрытую модель нельзя читать или восстанавливать обходными способами; агент должен узнавать поведение через пилоты.")
