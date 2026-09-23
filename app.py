"""Streamlit demo UI for the Beeline tariff campaign agent."""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
CAMPAIGN_COLUMNS = ["campaign_name", "filter_arpu_segment", "filter_data_segment",
                    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel"]
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

st.set_page_config(page_title="Enrique-agent | Beeline", page_icon="🟡", layout="centered")
st.markdown("""
<style>
    :root {color-scheme: dark;}
    .stApp {background:#0c0c0c; color:#f3f3f3;}
    .block-container {max-width:920px; padding-top:1.2rem; padding-bottom:4rem;}
    .brandbar {display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.45rem 0 1rem; border-bottom:1px solid #333; margin-bottom:1.8rem;}
    .brand-left {display:flex; align-items:center; gap:.65rem; font-size:1.04rem; font-weight:800; letter-spacing:.01em;}
    .brand-mark {width:32px; height:32px; flex:none;}
    .brand-line {width:1px; height:20px; background:#565656; margin:0 .2rem;}
    .brand-agent {font-weight:600; color:#d6d6d6;}
    .brand-slogan {font-size:.78rem; font-weight:650; color:#ffdb28; text-align:right;}
    .eyebrow {color:#ffdc24; font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.65rem;}
    .headline {font-size:clamp(2.15rem,5vw,3.8rem); font-weight:850; line-height:1.05; letter-spacing:-.045em; max-width:750px; margin:0 0 .8rem; color:#fff;}
    .headline em {font-style:normal; color:#ffdc24;}
    .lede {max-width:660px; font-size:1rem; line-height:1.55; color:#bdbdbd; margin-bottom:1.5rem;}
    .fact-strip {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:#353535; border:1px solid #353535; border-radius:13px; overflow:hidden; margin:0 0 2rem;}
    .fact {background:#171717; padding:.9rem 1rem; min-height:80px;}
    .fact strong {display:block; color:#ffdc24; font-size:1.35rem; line-height:1.15; letter-spacing:-.03em;}
    .fact span {color:#aaa; font-size:.76rem;}
    .section-tag {display:block; color:#ffdc24; font-size:.7rem; font-weight:800; text-transform:uppercase; letter-spacing:.15em; margin:0 0 .2rem;}
    .notice {background:#181818; border:1px solid #373737; border-left:3px solid #ffdc24; border-radius:9px; padding:.7rem .9rem; color:#c6c6c6; font-size:.84rem; line-height:1.45; margin:.6rem 0 1rem;}
    .notice b {color:#f2f2f2;}
    .explain-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:.6rem 0 1.5rem;}
    .explain-item {border-top:2px solid #ffdc24; background:#171717; padding:.8rem .9rem; min-height:90px;}
    .explain-item b {display:block; font-size:.9rem; margin-bottom:.35rem;}
    .explain-item span {color:#aaa; font-size:.78rem; line-height:1.35;}
    div[data-testid="stMetric"] {background:#181818; border:1px solid #343434; padding:13px; border-radius:10px;}
    div[data-testid="stMetric"] * {color:#f7f7f7 !important;}
    div[data-testid="stMetricLabel"] * {color:#aaa !important;}
    div[data-testid="stButton"] button[kind="primary"] {background:#ffdc24 !important; border-color:#ffdc24 !important; color:#141414 !important; font-weight:800; border-radius:8px;}
    div[data-testid="stButton"] button[kind="primary"]:hover {background:#ffe761 !important; border-color:#ffe761 !important;}
    div[data-testid="stDownloadButton"] button {border-radius:8px; border-color:#d7bc30; color:#ffdc24; font-weight:700;}
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:.15rem;}
    div[data-testid="stTabs"] [aria-selected="true"] {color:#ffdc24 !important;}
    div[data-testid="stDataFrame"] {border:1px solid #363636; border-radius:8px; overflow:hidden;}
    @media(max-width:680px) {.brand-slogan {display:none;} .fact-strip {grid-template-columns:repeat(2,minmax(0,1fr));} .explain-grid {grid-template-columns:1fr;} .explain-item {min-height:auto;}}
</style>
<div class="brandbar">
  <div class="brand-left">
    <svg class="brand-mark" viewBox="0 0 48 48" role="img" aria-label="Знак Билайн" xmlns="http://www.w3.org/2000/svg"><circle cx="24" cy="24" r="23" fill="#161616"/><path d="M3 15 C12 6 22 5 38 8" fill="none" stroke="#ffdc24" stroke-width="7"/><path d="M1 28 C14 17 29 20 46 17" fill="none" stroke="#ffdc24" stroke-width="8"/><path d="M9 40 C22 31 33 34 45 30" fill="none" stroke="#ffdc24" stroke-width="8"/></svg>
    <span>Билайн</span><span class="brand-line"></span><span class="brand-agent">Enrique-agent</span>
  </div><div class="brand-slogan">Жарқын жақта өмір сүр</div>
</div>
<div class="eyebrow">HackAlem AI / тарифные кампании</div>
<div class="headline">От пилота к <em>прибыльному решению.</em></div>
<div class="lede">Агент анализирует аудиторию, проверяет гипотезы на небольших группах и собирает план кампаний. Здесь можно показать его работу и проверить ограничения кейса.</div>
<div class="fact-strip">
  <div class="fact"><strong>23 441</strong><span>синтетических абонентов</span></div>
  <div class="fact"><strong>21</strong><span>доступный тарифный план</span></div>
  <div class="fact"><strong>≤ 20</strong><span>пилотных проверок</span></div>
  <div class="fact"><strong>100 000</strong><span>у.е. общего бюджета</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<span class="section-tag">01 / локальный запуск</span>', unsafe_allow_html=True)
st.subheader("Запустить агента")
st.markdown("""<div class="notice"><b>Локальная симуляция.</b> Демо использует мок-модель кейса.
Показанные суммы иллюстрируют механику скорера и не предсказывают результат на скрытых эффектах.</div>""", unsafe_allow_html=True)
input_col, button_col = st.columns([1, 1])
with input_col:
    run_count = st.number_input("Количество прогонов", min_value=1, max_value=50, value=1, step=1,
                                help="1 прогон — seed 42, несколько — seed 0, 1, 2 и так далее.")
with button_col:
    st.write("")
    run_button = st.button("Запустить агента", type="primary", use_container_width=True)
st.caption("Один прогон показывает решение; несколько проверяют устойчивость к шуму пилотов.")


def run_one(seed: int) -> dict:
    """Запускает Agent.act и тот же открытый скорер, что использует local_eval."""
    if not ARCHIVE.is_dir():
        raise FileNotFoundError("Не найдена папка archive рядом с app.py")
    previous_dir = Path.cwd()
    os.chdir(ARCHIVE)
    try:
        import agent as agent_module
        from mock_environment import make_mock_env, _mock_fallback, _mock_impact_model
        from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns, validate_strategy

        Agent = importlib.reload(agent_module).Agent
        env, internals = make_mock_env(seed=seed)
        agent_started = time.perf_counter()
        raw_plan = Agent().act(env) or []
        agent_seconds = time.perf_counter() - agent_started
        submission = pd.DataFrame(raw_plan)
        for column in CAMPAIGN_COLUMNS:
            if column not in submission:
                submission[column] = None
        submission = submission[CAMPAIGN_COLUMNS]
        valid_plan = sanitize_campaigns(raw_plan, env.tariffs)
        final_plan = valid_plan[:MAX_CAMPAIGNS]
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
        result["agent_seconds"] = agent_seconds
        result["rejected_campaigns"] = len(raw_plan) - len(valid_plan)
        try:
            validate_strategy(submission, env.tariffs)
            result["submission_error"] = None
        except ValueError as exc:
            result["submission_error"] = str(exc)
        return result
    finally:
        os.chdir(previous_dir)


if run_button:
    seeds = [42] if run_count == 1 else list(range(run_count))
    outcomes = []
    with st.spinner("Агент проводит пилоты и собирает план кампаний…"):
        progress = st.progress(0, text=f"Прогонов: 0 / {len(seeds)}")
        for index, seed in enumerate(seeds, start=1):
            try:
                outcomes.append(run_one(seed))
            except Exception as exc:
                outcomes.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            progress.progress(index / len(seeds), text=f"Прогонов: {index} / {len(seeds)}")
        progress.empty()
    st.session_state["demo_outcomes"] = outcomes

outcomes = st.session_state.get("demo_outcomes", [])
if not outcomes:
    st.markdown('<span class="section-tag">02 / что покажет демо</span>', unsafe_allow_html=True)
    st.subheader("Путь к решению")
    st.markdown("""<div class="explain-grid">
      <div class="explain-item"><b>1. Кандидаты</b><span>Профиль аудитории и история переходов дают гипотезы «сегмент → тариф».</span></div>
      <div class="explain-item"><b>2. Пилоты</b><span>Агент проверяет гипотезы через env.run_pilot() и учитывает шум измерения.</span></div>
      <div class="explain-item"><b>3. План</b><span>Выбирает канал и кампании с учётом чистого эффекта, бюджета и охвата.</span></div>
    </div>""", unsafe_allow_html=True)
    st.caption("Запусти один прогон для плана или несколько, чтобы увидеть, насколько решение меняется из-за шума пилотов.")
else:
    errors = [row for row in outcomes if row.get("error")]
    valid = [row for row in outcomes if not row.get("error")]
    for row in errors:
        st.error(f"Seed {row['seed']}: {row['error']}")

    if valid:
        focus = valid[0]
        st.markdown('<span class="section-tag">02 / результат симуляции</span>', unsafe_allow_html=True)
        st.subheader("Решение агента")
        money_cols = st.columns(3)
        money_cols[0].metric("Чистый результат, у.е.", f"{focus['net_arpu_gain']:,.0f}")
        money_cols[1].metric("Пилоты", f"{focus['n_pilots']} / 20")
        money_cols[2].metric("Итоговые кампании", f"{len(focus['plan_rows'])} / 10")
        st.caption(f"Seed {focus['seed']} · Прирост до затрат: {focus['gross_arpu_lift']:,.0f} у.е. · Расходы на контакты: {focus['total_cost']:,.0f} у.е.")
        st.progress(min(focus["budget_used_pct"] / 100, 1.0), text=f"Использовано бюджета: {focus['total_cost']:,.0f} / 100 000 у.е.")
        st.progress(min(focus["total_contacts"] / 15_000, 1.0), text=f"Использовано контактов: {focus['total_contacts']:,} / 15 000")
        st.markdown("**ARPU аудитории: до и после кампаний**")
        st.bar_chart(pd.DataFrame({"ARPU, у.е.": [focus["baseline_total_arpu"], focus["total_arpu_after"]]},
                                  index=["До кампаний", "После кампаний"]),
                     color="#ffdc24", use_container_width=True)
        st.caption(f"Изменение к базе: {focus['growth_vs_baseline_pct']:+.2f}%. После = до + прирост ARPU − стоимость контактов.")

        tab_plan, tab_economics, tab_pilots, tab_stability = st.tabs(
            ["План кампаний", "Экономика", "Пилоты", "Устойчивость"]
        )
        with tab_plan:
            plan = focus.get("plan_rows", pd.DataFrame())
            if plan.empty:
                st.warning("Агент не вернул итоговых кампаний.")
            else:
                st.markdown("**Кому → какой тариф → через какой канал**")
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
                details = details.loc[~details["name"].astype(str).str.startswith("pilot_")]
            if not details.empty:
                st.caption("Фактический охват и расходы по локальному скореру. Gross lift по строкам не складывается из-за дедупликации клиентов.")
                detail_columns = [c for c in ["name", "channel", "n_contacts", "cost", "gross_lift"] if c in details]
                st.dataframe(details[detail_columns].rename(columns={"name": "Кампания", "channel": "Канал",
                            "n_contacts": "Контакты", "cost": "Расход, у.е.", "gross_lift": "Gross lift, у.е."}),
                             use_container_width=True, hide_index=True)
        with tab_economics:
            st.markdown("**Как скорер получает чистый результат**")
            scored = pd.DataFrame(focus["campaigns_detail"])
            raw_gross = pd.to_numeric(scored["gross_lift"], errors="coerce").sum()
            unique_gross = float(focus["gross_arpu_lift"])
            overlap = raw_gross - unique_gross
            pilot_mask = scored["name"].astype(str).str.startswith("pilot_")
            pilot_cost = pd.to_numeric(scored.loc[pilot_mask, "cost"], errors="coerce").sum()
            final_cost = float(focus["total_cost"]) - pilot_cost
            bridge = pd.DataFrame([
                {"Шаг": "Сумма эффектов всех пилотов и кампаний", "Изменение, у.е.": raw_gross},
                {"Шаг": "Минус эффект повторно охваченных клиентов", "Изменение, у.е.": -overlap},
                {"Шаг": "Валовый прирост по уникальным клиентам", "Изменение, у.е.": unique_gross},
                {"Шаг": "Минус стоимость всех контактов", "Изменение, у.е.": -float(focus["total_cost"])},
                {"Шаг": "Чистый результат", "Изменение, у.е.": float(focus["net_arpu_gain"])},
            ])
            st.dataframe(bridge, use_container_width=True, hide_index=True,
                         column_config={"Изменение, у.е.": st.column_config.NumberColumn(format="%.0f")})
            st.caption("Сначала для каждого клиента оставляется лучший эффект; стоимость пилотов и повторных контактов вычитается полностью. Расчёт сделан по текущему прогону, без фиксированных чисел из README.")
            math_cols = st.columns(3)
            math_cols[0].metric("Уникальных клиентов", f"{focus['unique_customers_targeted']:,.0f}")
            math_cols[1].metric("Повторных контактов", f"{focus['total_contacts'] - focus['unique_customers_targeted']:,.0f}")
            math_cols[2].metric("Чистый эффект / клиент", f"{focus['net_arpu_gain'] / focus['unique_customers_targeted']:,.0f} у.е." if focus["unique_customers_targeted"] else "—")
            st.markdown("**Из чего состоят расходы**")
            spend = pd.DataFrame([
                {"Этап": "Пилоты", "Расход, у.е.": pilot_cost},
                {"Этап": "Итоговые кампании", "Расход, у.е.": final_cost},
            ])
            st.bar_chart(spend.set_index("Этап"), color="#ffdc24", use_container_width=True)
            by_channel = scored.groupby("channel", as_index=False).agg(
                contacts=("n_contacts", "sum"), cost=("cost", "sum")
            ).rename(columns={"channel": "Канал", "contacts": "Контакты", "cost": "Расход, у.е."})
            st.dataframe(by_channel, use_container_width=True, hide_index=True)
            st.caption("Таблица каналов включает пилоты и итоговые кампании. Их валовые эффекты по строкам нельзя просто складывать: аудитории могут пересекаться.")
            st.markdown("**Математика локальной модели**")
            st.code("эффект клиента = predicted_arpu × изменение ARPU при переходе\n"
                    "                × min(1, доля исторических переходов × множитель канала)\n"
                    "чистый результат = сумма лучших эффектов по уникальным клиентам\n"
                    "                 − стоимость всех контактов", language=None)
            st.caption("Это модельный прирост ARPU в локальной среде, а не реальные платежи. Историческая доля переходов не является измеренной вероятностью согласия клиента.")
        with tab_pilots:
            pilots = focus.get("pilot_rows", pd.DataFrame())
            if pilots.empty:
                st.info("Пилоты не запускались.")
            else:
                st.markdown("**Какие гипотезы агент проверил**")
                display = pilots.rename(columns={"pilot": "Пилот", "target_tariff": "Целевой тариф", "channel": "Канал",
                                                 "n_customers": "Клиенты", "cost": "Стоимость, у.е.",
                                                 "observed_lift_ratio": "Наблюдаемый эффект (шумный)",
                                                 "observed_lift_total": "Наблюдаемый прирост (шумный)",
                                                 "remaining_budget": "Остаток бюджета", "remaining_contacts": "Остаток контактов"})
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.caption("Пилот даёт зашумлённое наблюдение. Это сигнал для выбора кампаний, а не точная оценка будущего эффекта.")
                observed = pd.to_numeric(pilots["observed_lift_ratio"], errors="coerce").dropna()
                if not observed.empty:
                    histogram = pd.cut(observed, bins=8).value_counts().sort_index()
                    st.markdown("**Гистограмма наблюдаемых эффектов пилотов**")
                    st.bar_chart(pd.DataFrame({"Число пилотов": histogram.to_numpy()},
                                              index=[f"{band.left:.0%}…{band.right:.0%}" for band in histogram.index]),
                                 color="#ffdc24", use_container_width=True)
                    st.caption("Высота столбца показывает число пилотов в диапазоне; значения шумные и не равны истинному эффекту.")
        with tab_stability:
            if len(valid) > 1:
                nets = pd.Series([row["net_arpu_gain"] for row in valid], dtype=float)
                left, middle, right = st.columns(3)
                left.metric("Прогонов в плюс", f"{int((nets > 0).sum())} / {len(nets)}")
                middle.metric("Медиана, у.е.", f"{nets.median():,.0f}")
                right.metric("Худший прогон, у.е.", f"{nets.min():,.0f}")
                st.bar_chart(pd.DataFrame({"Чистый результат, у.е.": nets.to_numpy()},
                                          index=[f"Seed {row['seed']}" for row in valid]),
                             color="#ffdc24", use_container_width=True)
                st.dataframe(pd.DataFrame([{"seed": r["seed"], "net, у.е.": r["net_arpu_gain"],
                                            "пилотов": r["n_pilots"], "кампаний": len(r["plan_rows"])}
                                           for r in valid]), use_container_width=True, hide_index=True)
                st.caption("Разброс здесь связан с шумом пилотов в мок-среде, а не с неизвестной моделью судейства.")
            else:
                st.info("Для проверки разброса введи 10 прогонов и запусти агента снова. Организаторы предлагают этот режим для оценки устойчивости.")
        st.markdown('<span class="section-tag">03 / контроль решения</span>', unsafe_allow_html=True)
        with st.container():
            st.markdown("**Проверка ограничений на показанном прогоне**")
            checks = [
                ("Пилоты проведены", 0 < focus["n_pilots"] <= 20),
                ("Размер каждого пилота — 10–200", focus["pilot_rows"].empty or
                 focus["pilot_rows"]["n_customers"].between(10, 200).all()),
                ("Итоговых кампаний от 1 до 10", 1 <= len(focus["plan_rows"]) <= 10),
                ("Нет отброшенных кампаний", focus["rejected_campaigns"] == 0),
                ("Формат стратегии корректен", focus["submission_error"] is None),
                ("Бюджет ≤ 100 000 у.е.", focus["total_cost"] <= 100_000),
                ("Контакты ≤ 15 000", focus["total_contacts"] <= 15_000),
                ("Время Agent.act ≤ 10 минут", focus["agent_seconds"] <= 600),
            ]
            st.dataframe(pd.DataFrame([{"Требование": label, "Статус": "✓" if passed else "Проверить"}
                                       for label, passed in checks]), use_container_width=True, hide_index=True)
            if focus["submission_error"]:
                st.error(f"Ошибка формата: {focus['submission_error']}")
