# survey-tool

# AI-Powered Survey Platform 🧠

An intelligent, multi-modal survey creation and analysis platform built with Flask and powered by generative AI. This project goes beyond simple form building by incorporating a self-improving AI core, automated qualitative analytics, and multi-channel distribution capabilities.

---
## ✨ Key Features

This platform is designed with a focus on intelligent automation and data quality.

* **Self-Improving AI Core**: Utilizes a hybrid AI strategy. A custom-trained model is prioritized for survey generation, with a fallback to the Gemini API for new topics. New API results are cached and used to augment the dataset for future fine-tuning, creating a system that gets smarter and more cost-effective over time.
* **Multi-Modal Data Ingestion**: Collects responses from various sources:
    * **Digital Forms**: Standard, responsive web-based surveys.
    * **Paper Surveys (OCR)**: Automatically digitizes paper-based survey responses by processing uploaded images with Google Cloud Vision.
    * **Voice (IVR Foundation)**: Includes a foundational Interactive Voice Response system for telephone-based surveys.
* **Automated Qualitative Analytics**: Moves beyond simple charts by providing instant insights from text-based answers:
    * **Sentiment Analysis**: Automatically determines the sentiment (Positive/Negative) of open-ended responses.
    * **Theme Extraction**: Uses NLP to identify and display the most common keywords and themes.
* **Response Quality Flagging**: Automatically analyzes submission behavior and flags low-quality responses for:
    * **Speeding**: Completing the survey too quickly.
    * **Straight-lining**: Selecting the same answer for many questions in a row.
* **Integrated Multi-Channel Distribution**: Share surveys directly from the platform via:
    * Sharable Web Link & QR Code
    * SMS & WhatsApp (powered by Twilio)
* **Quiz Functionality**: Set correct answers for questions and automatically score results.
* **Multi-Language Support**: Create surveys in a default language, with respondents able to translate and submit in their preferred language.

---
## 🔧 Technology Stack

* **Backend**: Python, Flask, Flask-Login (for authentication)
* **Database**: SQLite
* **AI & Machine Learning**:
    * **Generative AI**: Google Gemini API, Fine-tuned Google Gemma 2B
    * **NLP & Analytics**: Hugging Face Transformers (`sentiment-analysis`), spaCy
    * **Fine-Tuning**: PEFT (Parameter-Efficient Fine-Tuning) with LoRA, TRL
    * **OCR**: Google Cloud Vision API
* **Services**: Twilio (for SMS, WhatsApp, and IVR)
* **Frontend**: HTML, CSS, JavaScript (rendered via Jinja2)

---
## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.9+
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up your environment variables:**
    * Create a file named `.env` in the project root.
    * Copy the contents of `.env.example` into it and fill in your secret keys and credentials.
    ```env
    # For Google Gemini API (Question Generation)
    GEMINI_API_KEY="your_gemini_api_key"

    # For Google Cloud Vision API (OCR Feature)
    GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"

    # For Twilio (SMS & WhatsApp)
    TWILIO_ACCOUNT_SID="your_twilio_account_sid"
    TWILIO_AUTH_TOKEN="your_twilio_auth_token"
    TWILIO_PHONE_NUMBER="your_twilio_sms_phone_number"
    TWILIO_WHATSAPP_NUMBER="your_twilio_whatsapp_number"
    
    # For Flask session security
    SECRET_KEY="a_very_long_and_random_secret_string"
    ```

6.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

---
## 🗺️ Future Roadmap

* Fully separate the frontend using a modern JavaScript framework like React or Vue.js.
* Complete the IVR system to allow full survey completion over the phone.
* Add more advanced analytics to the results dashboard (e.g., correlation analysis, data visualizations).
* Implement user roles and permissions (e.g., Admin, Editor, Viewer).

---
## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
