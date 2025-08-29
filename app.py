# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Observatorio de Ingresos",
    page_icon="📊",
    layout="wide"
)

# --- Funciones de Carga y Procesamiento (Cacheado) ---
@st.cache_data
def load_data(uploaded_file):
    """Carga y procesa los datos de ventas desde el archivo subido."""
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1', sep=';')
        df['FECHA DE EMISIÓN'] = pd.to_datetime(df['FECHA DE EMISIÓN'], dayfirst=True)
        numeric_cols = ['ÍTEM - CANTID TOTAL', 'VALOR - FACTURA DE VENTA']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        df['ÍTEM - CANTID TOTAL'] = df['ÍTEM - CANTID TOTAL'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
        return None

@st.cache_data
def run_market_basket_analysis(_df, min_support):
    """Prepara los datos y ejecuta el algoritmo Apriori."""
    basket = (_df.groupby(['FACTURA DE VENTA', 'ÍTEM - NOMBRE'])['ÍTEM - CANTID TOTAL']
              .sum().unstack().reset_index().fillna(0)
              .set_index('FACTURA DE VENTA'))
    def encode_units(x):
        return x >= 1
    basket_sets = basket.map(encode_units)
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    return frequent_itemsets

@st.cache_data
def run_prophet_forecast(_df_item_ts, periods_to_forecast, freq_code):
    """Ejecuta un pronóstico de Prophet para la serie de tiempo de un ítem."""
    df_prophet = _df_item_ts.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    m = Prophet()
    m.fit(df_prophet)
    
    prophet_freq = 'D' if freq_code == 'D' else 'W' if freq_code == 'W' else 'MS'
    
    future = m.make_future_dataframe(periods=periods_to_forecast, freq=prophet_freq)
    forecast = m.predict(future)
    return m, forecast

@st.cache_data
def train_projection_model(_df):
    """Prepara los datos a nivel mensual y entrena un modelo RandomForest para predecir ingresos."""
    df_sales_matrix = _df.pivot_table(
        index=pd.Grouper(key='FECHA DE EMISIÓN', freq='ME'),
        columns='ÍTEM - NOMBRE',
        values='ÍTEM - CANTID TOTAL',
        aggfunc='sum'
    ).fillna(0)
    
    df_revenue = _df.set_index('FECHA DE EMISIÓN')['VALOR - FACTURA DE VENTA'].resample('ME').sum()
    
    df_model_data = df_sales_matrix.join(df_revenue).dropna()
    
    if df_model_data.empty or 'VALOR - FACTURA DE VENTA' not in df_model_data.columns:
        return None, None, None, None, None, None

    X = df_model_data.drop('VALOR - FACTURA DE VENTA', axis=1)
    y = df_model_data['VALOR - FACTURA DE VENTA']
    
    if len(X) < 2:
        return None, None, None, None, None, None

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    baseline_scenario = X.mean()
    baseline_prediction = model.predict(pd.DataFrame([baseline_scenario]))[0]
    
    return model, X, baseline_scenario, baseline_prediction, importances, df_model_data

# --- PÁGINAS DE LA APLICACIÓN ---
def page_dashboard(df):
    st.title("📊 Observatorio de Ingresos: Dashboard General")
    st.markdown("Visión macro del rendimiento del negocio basada en datos transaccionales.")
    st.markdown("---")
    
    st.sidebar.header("Filtros del Dashboard")
    min_date = df['FECHA DE EMISIÓN'].min().date()
    max_date = df['FECHA DE EMISIÓN'].max().date()
    date_range = st.sidebar.date_input("Selecciona un rango de fechas:", (min_date, max_date), min_value=min_date, max_value=max_date, key="dashboard_date_range")
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df[(df['FECHA DE EMISIÓN'] >= start_date) & (df['FECHA DE EMISIÓN'] <= end_date)]

    st.header("Indicadores Clave de Rendimiento (KPIs)")
    total_revenue = df_filtered['VALOR - FACTURA DE VENTA'].sum()
    total_units_sold = df_filtered['ÍTEM - CANTID TOTAL'].sum()
    total_transactions = df_filtered['FACTURA DE VENTA'].nunique()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Ingresos Totales 💵", value=f"${total_revenue:,.2f}")
    kpi2.metric(label="Unidades Vendidas 📦", value=f"{int(total_units_sold):,}")
    kpi3.metric(label="Transacciones Totales 🧾", value=f"{total_transactions:,}")
    st.markdown("---")
    
    st.header("Análisis Visual")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Evolución de Ingresos")
        df_monthly_rev = df_filtered.set_index('FECHA DE EMISIÓN')['VALOR - FACTURA DE VENTA'].resample('ME').sum().reset_index()
        fig_rev = px.line(df_monthly_rev, x='FECHA DE EMISIÓN', y='VALOR - FACTURA DE VENTA', markers=True)
        st.plotly_chart(fig_rev, use_container_width=True)
    with col2:
        st.subheader("Top 10 Productos por Ingresos")
        top_revenue = df_filtered.groupby('ÍTEM - NOMBRE')['VALOR - FACTURA DE VENTA'].sum().nlargest(10).sort_values()
        fig_bar = px.bar(top_revenue, x='VALOR - FACTURA DE VENTA', y=top_revenue.index, orientation='h', text_auto='.2s')
        st.plotly_chart(fig_bar, use_container_width=True)


def page_market_basket(df):
    st.title("🧺 Análisis de la Cesta de Mercado")
    st.markdown("Descubre qué productos se compran juntos con frecuencia.")
    st.markdown("---")

    st.sidebar.header("Filtros de Asociación")
    min_support = st.sidebar.slider("Soporte Mínimo:", 0.001, 0.1, 0.005, 0.001, format="%.3f", help="Porcentaje de transacciones en las que aparece un conjunto de ítems.")
    min_confidence = st.sidebar.slider("Confianza Mínima:", 0.05, 1.0, 0.2, 0.05, format="%.2f", help="Probabilidad de comprar B si ya se compró A.")

    try:
        with st.spinner("Calculando reglas de asociación..."):
            frequent_itemsets = run_market_basket_analysis(df, min_support)
        
        if frequent_itemsets.empty:
            st.warning("No se encontraron conjuntos de ítems frecuentes. Intenta con un valor de soporte más bajo.")
        else:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            st.header("Reglas de Asociación Encontradas")
            st.markdown(f"Se encontraron **{len(rules)}** reglas con los filtros seleccionados.")

            if not rules.empty:
                sort_column = 'lift' if 'lift' in rules.columns else 'confidence'
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                display_cols = [col for col in display_cols if col in rules.columns]
                st.dataframe(rules[display_cols].sort_values(by=sort_column, ascending=False))
            else:
                st.warning("No se encontraron reglas de asociación. Intenta con una confianza más baja.")

    except Exception as e:
        st.error(f"Ocurrió un error durante el análisis: {e}")

def page_inventory_dynamics(df):
    st.title("📈 Dinámica y Pronóstico por Ítem")
    st.markdown("Analiza la tendencia, estacionalidad y pronóstico de ventas para productos individuales.")
    st.markdown("---")

    st.sidebar.header("Filtros de Dinámica")
    all_items = sorted(list(set(df['ÍTEM - NOMBRE'])))
    selected_item = st.sidebar.selectbox("Selecciona un producto:", all_items)
    
    freq_map = {'Diario': ('D', 90), 'Semanal': ('W', 52), 'Mensual': ('ME', 12)}
    selected_freq_label = st.sidebar.selectbox("Agrupar por:", freq_map.keys())
    selected_freq_code, forecast_periods = freq_map[selected_freq_label]

    if selected_item:
        st.header(f"Análisis para: {selected_item}")

        df_item_ts = df[df['ÍTEM - NOMBRE'] == selected_item]
        df_item_ts = df_item_ts.set_index('FECHA DE EMISIÓN')['ÍTEM - CANTID TOTAL'].resample(selected_freq_code).sum().fillna(0)
        
        st.subheader("Evolución de Unidades Vendidas")
        fig_ts = px.line(df_item_ts, x=df_item_ts.index, y='ÍTEM - CANTID TOTAL', markers=True, title=f"Ventas {selected_freq_label.lower()}s de {selected_item}")
        fig_ts.update_layout(xaxis_title='Fecha', yaxis_title='Unidades Vendidas')
        st.plotly_chart(fig_ts, use_container_width=True)

        with st.expander("Ver Descomposición de la Serie de Tiempo"):
            try:
                period_map = {'W': 52, 'ME': 12}
                period = period_map.get(selected_freq_code)
                
                if period and len(df_item_ts) > period * 2:
                    decomposition = seasonal_decompose(df_item_ts, model='additive', period=period)
                    fig_decomp = go.Figure()
                    fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Tendencia'))
                    fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Estacionalidad'))
                    fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residuos'))
                    fig_decomp.update_layout(title="Descomposición: Tendencia, Estacionalidad y Residuos", height=500)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                else:
                    st.warning(f"No hay suficientes datos ({len(df_item_ts)} puntos) para una descomposición estacional confiable a nivel {selected_freq_label.lower()}.")
            except Exception as e:
                st.error(f"Error en la descomposición: {e}")
        
        with st.expander("Ver Pronóstico de Demanda"):
            st.markdown("Utilizando **Prophet** para pronosticar la demanda futura de unidades.")
            freq_prophet_map = {'D': 'días', 'W': 'semanas', 'ME': 'meses'}
            periods_input = st.slider(f"Número de '{freq_prophet_map[selected_freq_code]}' a pronosticar:", 1, 104, forecast_periods)
            
            try:
                m, forecast = run_prophet_forecast(df_item_ts, periods_input, selected_freq_code)
                st.subheader("Gráfico del Pronóstico")
                fig_forecast = m.plot(forecast)
                st.pyplot(fig_forecast)
                st.subheader("Componentes del Pronóstico")
                fig_components = m.plot_components(forecast)
                st.pyplot(fig_components)
            except Exception as e:
                st.error(f"Ocurrió un error al generar el pronóstico: {e}")

def page_scenario_simulator(df):
    st.title("💡 Simulador de Escenarios de Ingresos")
    st.markdown("Ajusta las ventas mensuales de ítems clave y presiona 'Ejecutar' para ver el impacto en los ingresos.")
    st.markdown("---")
    
    with st.spinner("Entrenando modelo de proyección..."):
        model, features, baseline_scenario, baseline_prediction, importances, df_model_data = train_projection_model(df)
    
    if model is None:
        st.warning("No hay suficientes datos mensuales (se necesitan al menos 2 meses) para entrenar el modelo de simulación.")
        return

    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    with st.expander("Panel de Controles del Simulador", expanded=True):
        st.sidebar.header("Ajuste de Escenario")
        num_sliders = st.sidebar.slider("Número de ítems a ajustar:", 5, 25, 10)
        
        st.subheader(f"Ajusta las ventas mensuales para los {num_sliders} productos más importantes")
        
        slider_values = {}
        cols = st.columns(2)
        top_features = importances.head(num_sliders).index
        
        for i, item in enumerate(top_features):
            col = cols[i % 2]
            min_val = float(df_model_data[item].min())
            max_val = float(df_model_data[item].max())
            mean_val = float(df_model_data[item].mean())
            slider_max = max_val * 1.5 if max_val > 0 else 100
            
            slider_values[item] = col.slider(f"'{item}'", min_value=min_val, max_value=slider_max, value=mean_val, key=item)
        
        if st.button("Ejecutar Simulación 🚀", use_container_width=True):
            with st.spinner("Calculando predicción..."):
                current_scenario = baseline_scenario.copy()
                for item, value in slider_values.items():
                    current_scenario[item] = value
                
                df_current_scenario = pd.DataFrame([current_scenario])[features]
                current_prediction = model.predict(df_current_scenario)[0]
                
                st.session_state.prediction_result = current_prediction
    
    st.markdown("---")
    st.header("Resultados de la Simulación")

    if st.session_state.prediction_result is not None:
        current_prediction = st.session_state.prediction_result
        
        res1, res2 = st.columns(2)
        res1.metric(label="Ingreso Mensual Proyectado 📈", value=f"${current_prediction:,.2f}", delta=f"${current_prediction - baseline_prediction:,.2f} vs. Línea Base")
        res2.metric(label="Línea Base (Promedio Histórico) 📊", value=f"${baseline_prediction:,.2f}")
            
        df_chart = pd.DataFrame({'Escenario': ['Línea Base', 'Proyección Actual'], 'Ingresos': [baseline_prediction, current_prediction]})
        fig = px.bar(df_chart, x='Escenario', y='Ingresos', text_auto='.2s', title='Comparación de Escenarios')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ajusta los parámetros en el panel de controles y presiona 'Ejecutar Simulación' para ver los resultados.")

# --- Lógica Principal de la App ---
def main():
    st.sidebar.title("Navegación")
    page_options = ["Dashboard General", "Análisis de Cesta de Mercado", "Dinámica de Ítems", "Simulador de Escenarios"]
    page = st.sidebar.radio("Selecciona una página:", page_options)
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo de ventas (CSV)", type=["csv"], key="main_uploader")
    
    if uploaded_file is not None:
        df_main = load_data(uploaded_file)
        
        if df_main is not None:
            if page == "Dashboard General":
                page_dashboard(df_main)
            elif page == "Análisis de Cesta de Mercado":
                page_market_basket(df_main)
            elif page == "Dinámica de Ítems":
                page_inventory_dynamics(df_main)
            elif page == "Simulador de Escenarios":
                page_scenario_simulator(df_main)
    else:
        st.info("Por favor, sube tu archivo de datos para comenzar.")

if __name__ == "__main__":
    main()