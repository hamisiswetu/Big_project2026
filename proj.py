# app.py - Dashboard moja yenye models 3
# Endesha kwa:  streamlit run app.py

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. PAKIA MODELS 3 (hazifundishwi upya, tunazisoma tu)
# ---------------------------------------------------------
retail_model = joblib.load("online_retail_model.pkl")
grocery_model = joblib.load("model.joblib")
anomaly_model = joblib.load("anomaly_model.joblib")


# ---------------------------------------------------------
# 2. FUNCTION YA MAPENDEKEZO
# Inapokea: rules, bidhaa ambazo mteja anazo (basket), na lift ya chini
# Inarudisha: jedwali la bidhaa zinazopendekezwa
# ---------------------------------------------------------
def recommend(rules, basket, min_lift):
    results = []
    for row in rules.itertuples():                       # pitia rule moja moja
        if row.lift < min_lift:                          # ruka rule zenye lift ndogo
            continue
        if set(row.antecedents).issubset(basket):        # bidhaa za mteja zinalingana na rule?
            for product in row.consequents:
                if product not in basket:                # usipendekeze alichonacho tayari
                    results.append({
                        "Recommended Product": product,
                        "Because of": ", ".join(row.antecedents),
                        "Confidence": round(row.confidence, 2),
                        "Lift": round(row.lift, 2),
                    })
    table = pd.DataFrame(results)
    if table.empty:
        return table
    table = table.sort_values("Lift", ascending=False)               # lift kubwa juu
    return table.drop_duplicates("Recommended Product")              # kila bidhaa mara moja


# ---------------------------------------------------------
# 3. UKURASA WA MANUAL ENTRY (Retail na Grocery zinatumia huu huu)
# ---------------------------------------------------------
def manual_entry(model, min_lift):
    # Orodha ya bidhaa zote za model (mtumiaji anachagua tu)
    item = st.selectbox("Chagua bidhaa:", model["items"])

    # Kitufe cha kupata mapendekezo
    if st.button("Pata mapendekezo"):
        result = recommend(model["rules"], {item}, min_lift)

        if result.empty:
            st.info("Hakuna mapendekezo kwa lift hii. Jaribu kushusha slider.")
        else:
            st.dataframe(result)


# ---------------------------------------------------------
# 4. UKURASA WA UPLOAD FORM (Retail na Grocery zinatumia huu huu)
# col_customer, col_date, col_item = majina ya columns kwenye faili
# ---------------------------------------------------------
def upload_form(model, min_lift, col_customer, col_date, col_item):
    st.write("Faili liwe na columns:", col_customer, ",", col_date, ",", col_item)
    file = st.file_uploader("Pakia CSV", type="csv")
    if file is None:
        return                                           # hakuna faili bado

    data = pd.read_csv(file, encoding="latin1")
    data = data[[col_customer, col_date, col_item]].dropna()     # columns 3 tu, ondoa tupu
    data[col_item] = data[col_item].str.strip().str.lower()      # herufi ndogo
    data["Day"] = data[col_date].astype(str).str.split(" ").str[0]   # tarehe bila saa

    # Kapu moja = mteja mmoja + siku moja
    baskets = data.groupby([col_customer, "Day"])[col_item].apply(set)

    # Kuchanganua baskets zote ni polepole, kwa hiyo tunachukua idadi fulani tu
    limit = st.number_input("Idadi ya baskets za kuchanganua", 10, 5000, 200)
    baskets = baskets.head(limit)

    all_results = []
    for (customer, day), items in baskets.items():
        result = recommend(model["rules"], items, min_lift)
        result.insert(0, "Customer", customer)           # ongeza column ya mteja mwanzoni
        result.insert(1, "Date", day)
        all_results.append(result)

    final = pd.concat(all_results)                       # unganisha matokeo yote
    if final.empty:
        st.info("Hakuna mapendekezo kwa data hii.")
    else:
        st.write("Baskets zilizochanganuliwa:", len(baskets))
        st.dataframe(final)


# ---------------------------------------------------------
# 5. SIDEBAR (menu upande wa kushoto)
# ---------------------------------------------------------
st.sidebar.title("Dashboard")
section = st.sidebar.radio("Section", ["Home", "Online Retail", "Grocery", "Anomaly Detection"])

if section != "Home":
    page = st.sidebar.radio("Ukurasa", ["Manual Entry", "Upload Form"])

# ---------------------------------------------------------
# 6. MAIN CONTENT (upande wa kulia, unategemea ulichochagua)
# ---------------------------------------------------------
if section == "Home":
    st.title("Dashboard")
    st.write("Chagua section kwenye sidebar.")

elif section == "Online Retail":
    st.title("Online Retail - " + page)
    min_lift = st.sidebar.slider("Lift ya chini kabisa", 1.0, 20.0, 3.0, 0.1)
    if page == "Manual Entry":
        manual_entry(retail_model, min_lift)
    else:
        upload_form(retail_model, min_lift, "CustomerID", "InvoiceDate", "Description")

elif section == "Grocery":
    st.title("Grocery - " + page)
    min_lift = st.sidebar.slider("Lift ya chini kabisa", 1.0, 5.0, 1.2, 0.1)
    if page == "Manual Entry":
        manual_entry(grocery_model, min_lift)
    else:
        upload_form(grocery_model, min_lift, "Member_number", "Date", "itemDescription")

else:
    st.title("Anomaly Detection - " + page)

    if page == "Manual Entry":
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        if st.button("Angalia"):
            answer = anomaly_model.predict(pd.DataFrame({"Transaction_Amount": [amount]}))[0]
            if answer == 1:                              # 1 = Normal, -1 = Anomalous
                st.success("Normal")
            else:
                st.error("Anomalous")

    else:
        file = st.file_uploader("Pakia CSV au Excel", type=["csv", "xlsx"])
        if file is not None:
            if file.name.endswith(".csv"):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)
            data = data.rename(columns={"amount": "Transaction_Amount"})

            data["result"] = anomaly_model.predict(data[["Transaction_Amount"]])
            anomalies = data[data["result"] == -1]       # chukua zile zenye -1 tu
            st.write("Jumla ya transactions:", len(data), "| Anomalous:", len(anomalies))
            st.dataframe(anomalies)