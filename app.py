import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns

# For download buttons
from functionforDownloadButtons import download_button
import os
import json

# set page title and icon
st.set_page_config(
    page_title="BERT Keyword Extractor",
    # page_icon="🎈",
)

# set app layout width
def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()





with st.expander("About this app", expanded=True):

    st.write(
        """     
	    """
    )

    st.markdown("")



st.markdown("## ** Paste document **")


with st.form(key="my_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        # Model type
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)", "Flair"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Default (DistilBERT)":

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT(model=roberta)

            kw_model = load_model()

        else:

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=30,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 30.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",  value=True,
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )
    
    data = """We formalized a sustainability function to continue holding us accountable to the corporate governance standards that pillar responsible business. In summary, to ensure climate resilience, our Board, executives, and senior leadership monitor changing customer tastes and demands, regulatory requirements, as well as other impacts to our business. Our board oversight for ESG is a part of the Nominating and Corporate Governance Committees responsibility, and oversight specific to human capital and diversity, equity, and inclusion matters is delegated to the Human Capital and Compensation Committee. The ESG Committee meets regularly to address issues such as energy efficiency, carbon emissions, and environmental risks and opportunities in our homes and operations. The process to identify, manage, and integrate climate risk is embedded in our Enterprise Risk Assessment program, which is overseen by our Chief Financial Officer (CFO). The Human Capital and Compensation Committee of the board of trustees oversees our companys human capital programs and policies, including with respect to employee retention and development, and regularly meets with senior management to discuss these issues. This information is brought to our ESG Committee and discussed to help further develop our ESG strategy. Board oversight of ESG:, we formalized Board oversight for ESG as part of committee responsibilities, which we put into practice throughout. The Board is responsible for overseeing the companys approach to major risks and policies for assessing and managing these risks. The Nominating and Corporate Governance Committee has overall responsibility for our ESG program with specific topics overseen by the other Board committees. This council, led by members of senior management, provides participants with a unique forum to provide feedback from all levels in the organization. Quarterly cybersecurity reviews are led by our Chief Technology Officer and Vice President of Information Security, who leads our dedicated cybersecurity team, and other members of our executive leadership team, including our CEO, CFO, and CLO. In summary, to ensure climate resilience, our Board, executives, and senior leadership monitor changing customer tastes and demands, regulatory requirements, as well as other impacts to our business. we formalized a sustainability function to continue holding us accountable to the corporate governance standards that pillar responsible business. The Audit Committee and our Board also conduct a full review of cybersecurity annually. The Senior Vice President of Sustainability, AMH Development, and Property Operations teams collaborate to oversee the strategic implementation our environmental and energy programs. Board oversight of ESG:, we formalized Board oversight for ESG as part of committee responsibilities, which we put into practice throughout 2021. The Human Capital and Compensation Committee oversees our programs on talent, leadership and culture, which include diversity, equity and inclusion. Management role American Homes 4 Rents ESG Committee is responsible for supporting the companys efforts in developing, implementing, monitoring, and reporting on environmental initiatives including those relevant to climate change. 
     """,
    with c2:
        data = st.text_area(
            "Paste your text below (max 1000 words)",value="\n".join(data),
            height=510,
        )

        MAX_WORDS = 1000

        import re

        res = len(re.findall(r"\w+", data))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed"
            )

            data = data[:MAX_WORDS]

        submit_button = st.form_submit_button(label="Get me the data!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    data,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
    vectorizer=KeyphraseCountVectorizer()
)

st.markdown("## ** Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "🎁 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "🎁 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "🎁 Download (.json)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add Styling to the table columns and rows

cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
