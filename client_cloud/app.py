
import streamlit as st
import requests
import os
import requests
import pandas as pd
import os
API_URL = os.getenv("API_URL")

if not API_URL:
    st.error("API_URL environment variable is not set! The app cannot connect to the backend.")
    st.stop()

API_URL = os.getenv("API_URL", "http://localhost:8009")
# TRAINER_SERVER_URL = "http://model-trainer:8510"
CLS_SERVER_URL = API_URL

tab1, tab2 = st.tabs(["Upload Data","Make Prediction"])


with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Model from CSV URL")
        st.markdown("Enter a direct URL to a CSV file.")

        url = st.text_input("Enter CSV file URL")
        if st.button("Train Model from URL") :
            if url :
                with st.spinner("Training model...") :
                    try :
                        res = requests.post(f"{TRAINER_SERVER_URL}/train_from_url/", json={"url" : url})
                        res.raise_for_status()
                        data = res.json()
                        st.success(f"Model trained successfully! Accuracy: {data['accuracy']:.2%}")
                    except Exception as e :
                        st.error(f"Upload error: {e}")
            else :
                st.warning("Please enter a valid URL.")


    with col2 :
                st.subheader("Train Model from Local CSV")
                st.markdown("Upload a CSV file from your computer to train a new model.")

                uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

                if uploaded_file is not None :
                    try :
                        df_preview = pd.read_csv(uploaded_file)
                        st.success("File preview loaded successfully!")
                        st.markdown("### Preview:")
                        st.dataframe(df_preview.head())


                        uploaded_file.seek(0)

                        if st.button("Upload and Train Model", key="upload_train") :
                            with st.spinner("Uploading and training...") :

                                try :
                                    file_bytes = uploaded_file.getvalue()
                                    files = {"file" : (uploaded_file.name, file_bytes, "text/csv")}

                                    res = requests.post(f"{TRAINER_SERVER_URL}/train_from_upload/", files=files)

                                    res.raise_for_status()

                                    data = res.json()
                                    st.success(f"Model trained successfully! Accuracy: {data['accuracy']:.2%}")

                                except requests.exceptions.RequestException as e :
                                    error_message = f"A network or server error occurred: {e}"

                                    if e.response is not None :
                                        try :
                                            error_detail = e.response.json().get('detail', e.response.text)
                                            error_message = f"Error from server: {error_detail}"
                                        except ValueError :
                                            error_message = f"Server returned a non-JSON error ({e.response.status_code}): {e.response.text}"

                                    st.error(error_message)

                    except Exception as e :
                        st.error(
                            f"Error reading file for preview: Please ensure the CSV is well-formatted. Details: {e}")
with tab2:
            st.subheader("Make Prediction on Existing Model")
            st.markdown("Select a model and provide feature values for prediction.")

            # Load model columns
            col_response = requests.get(f"{CLS_SERVER_URL}/model_columns/")
            if col_response.status_code == 200:
                columns_dict = col_response.json()

                input_data = {}
                st.markdown("### Feature Values:")
                for col, options in columns_dict.items():
                    input_data[col] = st.selectbox(f"{col}", options)

                if st.button("Predict"):
                    with st.spinner("Making prediction..."):
                        pred_response = requests.post(f"{CLS_SERVER_URL}/predict/", json={
                            "features": input_data
                        })
                        if pred_response.status_code == 200:
                            prediction = pred_response.json()["prediction"]
                            st.success(f"Prediction: {prediction}")
                        else:
                            st.error(f"Prediction error: {pred_response.text}")
            else:
                st.error("Failed to load model columns.")


