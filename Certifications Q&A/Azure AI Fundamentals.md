# **AI-900**

<details>
  <summary><strong>Q1: What is supervised machine learning?</strong></summary>
  <p>Supervised machine learning is a type of ML where the training data includes both input features and known labels. The model learns the relationship between them to predict labels for future observations.</p>
</details>

<details>
  <summary><strong>Q2: What are the two main types of supervised machine learning tasks?</strong></summary>
  <p>Regression and Classification.</p>
</details>

<details>
  <summary><strong>Q1. What are the two types of Azure AI service resources you can create?</strong></summary>
  <ul>
    <li>Multi-service resource</li>
    <li>Single-service resource</li>
  </ul>
</details>

<details>
  <summary><strong>Q2. What is a multi-service Azure AI resource used for?</strong></summary>
  <p>A multi-service resource provides access to multiple Azure AI services with a single key and endpoint. It is useful when you need several AI services or are exploring AI capabilities.</p>
</details>

<details>
  <summary><strong>Q3. When should you consider using a single-service resource instead of a multi-service resource?</strong></summary>
  <ul>
    <li>You only require one AI service (e.g., Language, Speech, Face)</li>
    <li>You want to manage billing and cost information separately for that service</li>
  </ul>
</details>

<details>
  <summary><strong>Q4. How are Azure AI services billed when using a multi-service resource?</strong></summary>
  <p>All AI services accessed through a multi-service resource are billed together, under a single pricing plan.</p>
</details>

<details>
  <summary><strong>Q5. What information must be provided when creating an Azure AI service resource in the Azure portal?</strong></summary>
  <ul>
    <li>Subscription</li>
    <li>Resource Group</li>
    <li>Region</li>
    <li>Resource Name (must be unique)</li>
    <li>Pricing Tier</li>
  </ul>
</details>

<details>
  <summary><strong>Q6. How do you create a multi-service AI resource in the Azure portal?</strong></summary>
  <ol>
    <li>Sign in to the Azure portal with Contributor access.</li>
    <li>Select Create a resource.</li>
    <li>Search for Azure AI services in the marketplace.</li>
    <li>Select and configure the resource.</li>
  </ol>
</details>

<details>
  <summary><strong>Q7. How do you create a single-service AI resource in the Azure portal?</strong></summary>
  <ol>
    <li>Sign in to the Azure portal.</li>
    <li>Select Create a resource.</li>
    <li>Search for the specific AI service, such as Face, Language, or Content Safety.</li>
    <li>Select and configure the resource.</li>
  </ol>
</details>

<details>
  <summary><strong>Q8. Do Azure AI services have free pricing tiers?</strong></summary>
  <p>Yes, most Azure AI services offer a free price tier that allows you to explore and evaluate capabilities at no cost.</p>
</details>

## 📘 Grounding and Retrieval-Augmented Generation (RAG)

<details> <summary><strong>Q1. What is the purpose of grounding in generative AI systems?</strong></summary> A. Grounding ensures that a model's outputs are aligned with factual, contextual, or reliable data sources. It helps improve trustworthiness by anchoring responses to real-world information. </details> <details> <summary><strong>Q2. Which technique connects a language model to an organization’s proprietary data to generate more accurate responses?</strong></summary> A. Retrieval-Augmented Generation (RAG) </details> <details> <summary><strong>Q3. What is the primary benefit of using Retrieval-Augmented Generation (RAG) in AI systems?</strong></summary> A. RAG enhances a model’s performance by using real-time or domain-specific information, leading to more accurate, relevant, and up-to-date responses. </details> <details> <summary><strong>Q4. In which scenarios is RAG particularly useful?</strong></summary> A. RAG is ideal for applications requiring access to dynamic or proprietary data, such as customer support and knowledge management systems. </details>

## 📘Fine-Tuning and Security
<details> <summary><strong>Q5. What is fine-tuning in the context of generative AI?</strong></summary> A. Fine-tuning involves further training a pre-trained model on a task-specific or domain-specific dataset to improve its performance for specialized applications. </details> <details> <summary><strong>Q6. Why is fine-tuning useful for generative AI models?</strong></summary> A. It improves task-specific accuracy and reduces irrelevant or inaccurate responses by adapting the model to domain-specific needs. </details> <details> <summary><strong>Q7. What role do security and governance controls play in generative AI applications?</strong></summary> A. They manage access, authentication, and data usage to help prevent the generation or publication of incorrect or unauthorized information. </details>

## 📘 Evaluators and Quality Metrics
<details> <summary><strong>Q8. What are the three primary categories of evaluators used to measure generative AI response quality?</strong></summary> A. 1. Performance and quality evaluators 2. Risk and safety evaluators 3. Custom evaluators </details> <details> <summary><strong>Q9. What do performance and quality evaluators assess in AI-generated content?</strong></summary> A. They assess accuracy, groundedness, and relevance of the generated output. </details> <details> <summary><strong>Q10. What is the purpose of risk and safety evaluators in generative AI systems?</strong></summary> A. They evaluate potential risks such as harmful, biased, or inappropriate content to ensure safer outputs. </details> <details> <summary><strong>Q11. What are custom evaluators used for in generative AI quality measurement?</strong></summary> A. Custom evaluators are designed for industry-specific needs and use domain-specific metrics to align AI outputs with business goals. </details> <details> <summary><strong>Q12. Which Azure tool provides an environment for workflows like evaluation and responsible AI planning?</strong></summary> A. Azure AI Foundry </details>

## 📘Microsoft's Responsible Generative AI Process
<details> <summary><strong>Q1. What are the four stages in Microsoft’s responsible generative AI process?</strong></summary> A. 1. Identify potential harms 2. Measure the presence of harms in model outputs 3. Mitigate harms across solution layers and ensure transparent communication 4. Operate responsibly with deployment and operational readiness plans </details> <details> <summary><strong>Q2. What should inform the four stages of responsible AI planning?</strong></summary> A. Microsoft's Responsible AI principles </details> <details> <summary><strong>Q3. Why is grounding AI systems in responsible principles especially important?</strong></summary> A. Because AI systems are probabilistic, data-driven, and can influence high-stakes decisions, leading to potential harm if not carefully designed and monitored. </details>

## 📘Responsible AI Principles
<details> <summary><strong>Q4. What does the principle of Fairness in responsible AI emphasize?</strong></summary> A. That AI systems should treat all people fairly, without bias based on gender, ethnicity, or other discriminatory factors. </details> <details> <summary><strong>Q5. What is a good practice to ensure fairness during AI system development?</strong></summary> A. • Use representative training data • Continuously evaluate model performance across subpopulations </details> <details> <summary><strong>Q6. What does Reliability and Safety mean in the context of AI?</strong></summary> A. AI systems must function as expected and safely under all conditions, particularly in high-risk domains like healthcare or autonomous driving. </details> <details> <summary><strong>Q7. How can developers ensure reliability and safety in AI systems?</strong></summary> A. Through rigorous testing, confidence thresholding, and robust deployment practices. </details> <details> <summary><strong>Q8. What does the Privacy and Security principle involve?</strong></summary> A. Protecting personal and sensitive data in training, deployment, and inference phases, and ensuring appropriate safeguards are in place. </details> <details> <summary><strong>Q9. How does Microsoft define Inclusiveness in AI systems?</strong></summary> A. AI should empower all users, including those from diverse backgrounds or with disabilities, by ensuring design and testing include broad user perspectives. </details> <details> <summary><strong>Q10. What does Transparency mean in responsible AI?</strong></summary> A. Users should understand how the system works, its purpose, limitations, and what influences its predictions—like features, training size, or confidence scores. </details> <details> <summary><strong>Q11. What is a key aspect of achieving transparency in AI systems?</strong></summary> A. Providing clear information about data usage, model limitations, and how predictions are made. </details> <details> <summary><strong>Q12. What is the core idea behind the Accountability principle?</strong></summary> A. Humans, not the system, are accountable for decisions made by AI. Developers and organizations must ensure AI systems meet ethical and legal standards. </details> <details> <summary><strong>Q13. Why must organizations build AI solutions within a governance framework?</strong></summary> A. To ensure AI applications comply with legal, ethical, and organizational responsibility standards. </details> <details> <summary><strong>Q14. Why is it dangerous for users to overly trust AI systems?</strong></summary> A. Because the human-like behavior of AI can lead to misplaced trust, even though the model might still make incorrect or biased predictions. </details>

## 📘 Basics of Text Analysis and NLP

<details>
  <summary><strong>Q1: What is the primary purpose of tokenization in text analysis?</strong></summary>
  <p>Tokenization breaks a text into smaller components called tokens, typically words or subwords, which serve as the base units for further analysis.</p>
</details>

<details>
  <summary><strong>Q2: What are stop words, and why are they removed in text analysis?</strong></summary>
  <p>Stop words are common words (like “the”, “a”, “it”) that add little semantic meaning. Removing them improves the signal-to-noise ratio in NLP tasks.</p>
</details>

<details>
  <summary><strong>Q3: Define n-grams and list the types with examples.</strong></summary>
  <p>N-grams are contiguous sequences of n items from text.  
  - <strong>Unigram</strong>: "moon"  
  - <strong>Bigram</strong>: "to go"  
  - <strong>Trigram</strong>: "choose to go"</p>
</details>

<details>
  <summary><strong>Q4: What is stemming and why is it used in text processing?</strong></summary>
  <p>Stemming reduces words to their base/root form (e.g., “power”, “powered” → “power”) to consolidate similar words during analysis.</p>
</details>

---

## 📘Frequency Analysis

<details>
  <summary><strong>Q5: How does frequency analysis help in identifying the subject of a document?</strong></summary>
  <p>By identifying the most frequent (non-stop) words or phrases, one can infer the key topics or themes of the document.</p>
</details>

<details>
  <summary><strong>Q6: What is TF-IDF and why is it better than simple frequency analysis?</strong></summary>
  <p>TF-IDF (Term Frequency-Inverse Document Frequency) scores words based on how often they appear in a document versus across all documents. It helps highlight document-specific important terms.</p>
</details>

---

## 📘 Text Classification and Sentiment Analysis

<details>
  <summary><strong>Q7: What is text classification in NLP?</strong></summary>
  <p>It is the process of assigning predefined categories (e.g., sentiment: positive/negative) to text based on its content.</p>
</details>

<details>
  <summary><strong>Q8: How can machine learning be used for sentiment analysis?</strong></summary>
  <p>By training a classification model (e.g., logistic regression) using labeled text (e.g., reviews with sentiment scores), the model learns to predict sentiment from new text.</p>
</details>

<details>
  <summary><strong>Q9: Give examples of positive and negative sentiment tokens.</strong></summary>
  <p><strong>Positive</strong>: “great”, “tasty”, “fun”  
  <strong>Negative</strong>: “terrible”, “slow”, “substandard”</p>
</details>

---

## 📘 Embeddings and Semantic Models

<details>
  <summary><strong>Q10: What are embeddings in NLP?</strong></summary>
  <p>Embeddings are numerical vector representations of words that capture their semantic meaning and relationships in a multi-dimensional space.</p>
</details>

<details>
  <summary><strong>Q11: What does the closeness of embeddings in vector space signify?</strong></summary>
  <p>It indicates semantic similarity; closely located tokens often have related meanings.</p>
</details>

---

## 📘 Azure AI Language Features

<details>
  <summary><strong>Q12: What is Azure AI Language?</strong></summary>
  <p>It’s a service offered by Microsoft for performing advanced natural language processing tasks on unstructured text.</p>
</details>

<details>
  <summary><strong>Q13: List at least 5 capabilities of Azure AI Language.</strong></summary>
  <p>- Named Entity Recognition  
  - Entity Linking  
  - Language Detection  
  - Sentiment Analysis and Opinion Mining  
  - Summarization  
  - Key Phrase Extraction</p>
</details>

<details>
  <summary><strong>Q14: What is entity linking and how does it help?</strong></summary>
  <p>It connects detected entities to a knowledge base like Wikipedia to disambiguate and enrich entity information.</p>
</details>

<details>
  <summary><strong>Q15: What does the language detection feature return?</strong></summary>
  <p>- Language name (e.g., English)  
  - ISO 639-1 code (e.g., “en”)  
  - Confidence score</p>
</details>

<details>
  <summary><strong>Q16: How does Azure AI Language handle mixed-language text in language detection?</strong></summary>
  <p>It identifies the predominant language based on the majority of content and returns that along with a confidence score.</p>
</details>

<details>
  <summary><strong>Q17: What happens if the input text is ambiguous or contains only symbols?</strong></summary>
  <p>The service may return the language name as “unknown”, the identifier as “unknown”, and the score as NaN.</p>
</details>

---

## 📘 Entity Recognition and Text Summarization

<details>
  <summary><strong>Q18: What is Named Entity Recognition (NER)?</strong></summary>
  <p>NER identifies and classifies entities like people, locations, organizations, dates, and quantities in text.</p>
</details>

<details>
  <summary><strong>Q19: How is Entity Linking different from NER?</strong></summary>
  <p>While NER identifies named entities, Entity Linking connects these entities to a reference knowledge base (e.g., Wikipedia) for disambiguation and added context.</p>
</details>

<details>
  <summary><strong>Q20: What is extractive summarization and how does Azure implement it?</strong></summary>
  <p>Extractive summarization pulls the most important sentences from a document to create a summary. Azure AI Language uses models to identify and rank these key sentences.</p>
</details>

---

## 📘 Opinion Mining and Sentiment Scores

<details>
  <summary><strong>Q21: What is opinion mining?</strong></summary>
  <p>Opinion mining is a finer-grained form of sentiment analysis that detects sentiment towards specific aspects or targets within text.</p>
</details>

<details>
  <summary><strong>Q22: What does a sentiment score represent in Azure AI Language?</strong></summary>
  <p>It's a confidence value (from 0 to 1) indicating the likelihood of a sentence or document being positive, neutral, or negative in sentiment.</p>
</details>

<details>
  <summary><strong>Q23: How can sentiment analysis help businesses?</strong></summary>
  <p>It helps in understanding customer opinions at scale, improving products, customer service, and brand perception.</p>
</details>

---

## 📘 Text Preprocessing Techniques

<details>
  <summary><strong>Q24: Why is text normalization important in NLP?</strong></summary>
  <p>Normalization (like lowercasing, removing punctuation) ensures consistency, improving the accuracy of downstream NLP tasks.</p>
</details>

<details>
  <summary><strong>Q25: What is the difference between stemming and lemmatization?</strong></summary>
  <p>- <strong>Stemming</strong> cuts off word endings roughly (e.g., “running” → “run”).  
  - <strong>Lemmatization</strong> reduces words to their base form using vocabulary and grammar (e.g., “better” → “good”).</p>
</details>

---

## 📘 Word Similarity and Contextual Understanding

<details>
  <summary><strong>Q26: How does word embedding help in measuring similarity?</strong></summary>
  <p>It converts words into vectors where similar words are located near each other in high-dimensional space, allowing comparison using metrics like cosine similarity.</p>
</details>

<details>
  <summary><strong>Q27: Give an example of how context affects word meaning in NLP.</strong></summary>
  <p>The word "bank" in "river bank" vs. "savings bank" has different meanings. Contextual embeddings like BERT capture these nuances.</p>
</details>

---

## 📘 Practical Considerations in Azure AI Language

<details>
  <summary><strong>Q28: What format must text data be in to use Azure AI Language APIs?</strong></summary>
  <p>Input must typically be structured as JSON with fields like documents, each containing an id, language, and text.</p>
</details>

<details>
  <summary><strong>Q29: Can Azure AI Language services process documents in languages other than English?</strong></summary>
  <p>Yes, it supports multiple languages, though capabilities may vary by language.</p>
</details>

<details>
  <summary><strong>Q30: What should you do if Azure AI Language returns “unknown” as the detected language?</strong></summary>
  <p>Verify that the text has enough meaningful characters and avoid special characters or extremely short phrases.</p>
</details>

## 📘 Fundamentals of question answering with the Language Service
<details>
  <summary><strong>Q1. What is the purpose of Question Answering in Azure AI Language?</strong></summary>
  <p>It enables natural language AI workloads by allowing users to get accurate, conversational answers from a knowledge base, often used in bots on platforms like websites or social media.</p>
</details>

<details>
  <summary><strong>Q2. What are typical use cases for Question Answering solutions?</strong></summary>
  <p>
    • Automating responses to customer queries<br>
    • Providing 24/7 self-service support<br>
    • Enhancing chatbots with natural multi-turn conversations<br>
    • Improving user engagement on websites or social platforms
  </p>
</details>

<details>
  <summary><strong>Q3. How does Azure help create a Question Answering solution?</strong></summary>
  <p>Azure AI Language provides a custom question answering feature through Language Studio, allowing users to build a knowledge base from FAQ documents or manual entries.</p>
</details>

<details>
  <summary><strong>Q4. What types of sources can be used to define question-and-answer pairs in Azure AI Language?</strong></summary>
  <p>
    • Existing FAQ documents<br>
    • Web pages<br>
    • Manually entered questions and answers<br>
    • A combination of all the above
  </p>
</details>

<details>
  <summary><strong>Q5. What tool can you use to create and manage Question Answering projects in Azure?</strong></summary>
  <p>Language Studio, a web-based interface for managing Azure AI Language projects.</p>
</details>

<details>
  <summary><strong>Q6. What happens when you save your question-and-answer project in Language Studio?</strong></summary>
  <p>Azure processes the questions and answers using a natural language model that enables it to return correct answers even if the user's phrasing differs from the original question.</p>
</details>

<details>
  <summary><strong>Q7. How can you test your custom Question Answering project?</strong></summary>
  <p>Use the built-in test interface in Language Studio by submitting natural language queries and reviewing the returned answers.</p>
</details>

<details>
  <summary><strong>Q8. Why is Question Answering considered user-friendly?</strong></summary>
  <p>It allows users to ask questions in natural language and receive instant, relevant answers — at any time, not just during business hours.</p>
</details>

<details>
  <summary><strong>Q9. What key benefit does multi-turn conversation provide in Question Answering bots?</strong></summary>
  <p>It enables follow-up questions and more natural, human-like dialogue flow in bot interactions.</p>
</details>

<details>
  <summary><strong>Q10. What must be done before testing a Question Answering project in Azure?</strong></summary>
  <p>The project must be saved, which triggers NLP processing to map varied user inputs to the right answers.</p>
</details>

## 📘 Fundamentals of conversational language understanding

<details>
  <summary><strong>Q1. What is the purpose of Azure AI Language’s Conversational Language Understanding (CLU) feature?</strong></summary>
  <p>CLU allows you to create and use natural language models that identify intents and entities from user input, enabling apps like chatbots to understand and respond to human language.</p>
</details>

<details>
  <summary><strong>Q2. What are the three main components you define when authoring a CLU model?</strong></summary>
  <p>
    • Intents – The goal or purpose behind the user's input<br>
    • Entities – Specific data or information in the input (e.g., dates, names)<br>
    • Utterances – Example phrases that express an intent
  </p>
</details>

<details>
  <summary><strong>Q3. What is the purpose of training a CLU model?</strong></summary>
  <p>Training teaches the model to understand which intents and entities correspond to user utterances so it can make accurate predictions.</p>
</details>

<details>
  <summary><strong>Q4. Which Azure resource allows both authoring and prediction for CLU?</strong></summary>
  <p>The Azure AI Language resource.</p>
</details>

<details>
  <summary><strong>Q5. Can you use Azure AI Services resource for authoring CLU models?</strong></summary>
  <p>No, Azure AI Services can only be used for prediction, not authoring.</p>
</details>

<details>
  <summary><strong>Q6. Why is the separation between Azure AI Language and Azure AI Services resources useful?</strong></summary>
  <p>It allows you to track usage separately for authoring and client-side prediction workloads.</p>
</details>

<details>
  <summary><strong>Q7. How do you publish a CLU model for client applications to use?</strong></summary>
  <p>After training and testing, you publish the model to a prediction resource, which provides an endpoint for applications to connect and use.</p>
</details>

<details>
  <summary><strong>Q8. How do client applications interact with a trained CLU model?</strong></summary>
  <p>By sending user input to the prediction resource endpoint using the correct authentication key, and receiving predicted intents and entities in response.</p>
</details>

<details>
  <summary><strong>Q9. What interface can be used to create and manage CLU models easily without writing code?</strong></summary>
  <p>Language Studio – a web-based interface provided by Azure.</p>
</details>

<details>
  <summary><strong>Q10. What are prebuilt domains in CLU, and how can they help?</strong></summary>
  <p>Prebuilt domains offer ready-made intents and entities for common scenarios (e.g., calendar, weather), allowing faster and easier model creation.</p>
</details>

<details>
  <summary><strong>Q11. Can entities and intents be created in any order?</strong></summary>
  <p>Yes, you can define either first and then map them during the authoring process.</p>
</details>

<details>
  <summary><strong>Q12. Why is training and testing a CLU model considered an iterative process?</strong></summary>
  <p>Because you may need to adjust sample utterances, intents, or entities based on test results, retrain the model, and re-test to improve accuracy.</p>
</details>

<details>
  <summary><strong>Q13. What is an intent in a CLU model?</strong></summary>
  <p>An intent represents the purpose or goal behind a user's input (e.g., “BookFlight”, “CheckWeather”).</p>
</details>

<details>
  <summary><strong>Q14. What is an entity in a CLU model?</strong></summary>
  <p>An entity is a specific piece of information extracted from the user's input (e.g., a date, location, or name).</p>
</details>

<details>
  <summary><strong>Q15. What are utterances used for in CLU?</strong></summary>
  <p>Utterances are example phrases that users might say to express a particular intent, used to train the model.</p>
</details>

<details>
  <summary><strong>Q16. What happens when you publish a CLU model?</strong></summary>
  <p>Publishing makes the model available through a prediction endpoint, allowing applications to send input and receive predictions.</p>
</details>

<details>
  <summary><strong>Q17. Can CLU models be trained with user-defined intents and entities?</strong></summary>
  <p>Yes, users can define custom intents and entities to suit their application's needs.</p>
</details>

<details>
  <summary><strong>Q18. How does Azure ensure that different expressions can match the same intent?</strong></summary>
  <p>It uses machine learning models trained on various utterances to understand paraphrased or similar expressions.</p>
</details>

<details>
  <summary><strong>Q19. What is the main advantage of using Language Studio to author CLU models?</strong></summary>
  <p>It provides a code-free, visual interface that simplifies the creation, training, and testing of models.</p>
</details>

<details>
  <summary><strong>Q20. Why is it important to test a CLU model before publishing?</strong></summary>
  <p>To ensure the model accurately identifies the correct intents and entities, improving performance and user experience.</p>
</details>

<details>
  <summary><strong>Q21. What kind of applications can benefit from CLU?</strong></summary>
  <p>Applications like chatbots, virtual agents, and automated customer support systems that require natural language understanding.</p>
</details>

<details>
  <summary><strong>Q22. Which step comes immediately after training a CLU model?</strong></summary>
  <p>Testing the model with sample utterances to evaluate its performance.</p>
</details>

<details>
  <summary><strong>Q23. What do client applications need to access the CLU model endpoint?</strong></summary>
  <p>The endpoint URL and an authentication key for the prediction resource.</p>
</details>

<details>
  <summary><strong>Q24. What happens if your CLU model does not recognize the correct intent during testing?</strong></summary>
  <p>You should revise the utterances, add more examples, retrain, and test again—it’s an iterative improvement process.</p>
</details>

<details>
  <summary><strong>Q25. What is a key benefit of using prebuilt domains in CLU?</strong></summary>
  <p>They reduce setup time by providing ready-to-use intents and entities for common business scenarios.</p>
</details>

## 📘 Azure AI speech
<details>
  <summary><strong>Q1: What is the main function of speech recognition in Azure AI Speech?</strong></summary>
  <p>To convert spoken audio into text data that applications can process.</p>
</details>

<details>
  <summary><strong>Q2: What are the two core models used in speech recognition?</strong></summary>
  <p>• Acoustic model – converts audio signals into phonemes. <br>
  • Language model – maps phonemes to words using statistical predictions.</p>
</details>

<details>
  <summary><strong>Q3: Give two real-world scenarios where speech-to-text is used.</strong></summary>
  <p>• Creating live captions for videos or events. <br>
  • Transcribing meetings or phone calls automatically.</p>
</details>

<details>
  <summary><strong>Q4: What does speech synthesis do?</strong></summary>
  <p>Converts text data into speech, enabling applications to "speak" text aloud.</p>
</details>

<details>
  <summary><strong>Q5: What parameters can be controlled in speech synthesis output?</strong></summary>
  <p>• Voice <br> 
  • Speaking rate <br>
  • Pitch <br>
  • Volume</p>
</details>

<details>
  <summary><strong>Q6: Name one difference between real-time and batch transcription.</strong></summary>
  <p>• Real-time transcription is used for live audio input. <br>
  • Batch transcription processes pre-recorded audio files asynchronously.</p>
</details>

<details>
  <summary><strong>Q7: What type of model powers Azure’s speech-to-text API?</strong></summary>
  <p>The Universal Language Model, developed and trained by Microsoft.</p>
</details>

<details>
  <summary><strong>Q8: When should you use batch transcription over real-time transcription?</strong></summary>
  <p>When you have stored audio files and do not require instant results.</p>
</details>

<details>
  <summary><strong>Q9: What kind of audio input can real-time transcription accept?</strong></summary>
  <p>Live microphone input or streamed audio from an audio file.</p>
</details>

<details>
  <summary><strong>Q10: What does Azure’s text-to-speech feature enable you to do?</strong></summary>
  <p>Convert written text into spoken audio, which can be played or saved.</p>
</details>

<details>
  <summary><strong>Q11: What is a neural voice in Azure text-to-speech?</strong></summary>
  <p>A type of voice that uses neural networks to create natural-sounding speech with realistic intonation.</p>
</details>

<details>
  <summary><strong>Q12: Can you create custom voices with Azure AI Speech?</strong></summary>
  <p>Yes, developers can build and deploy custom voices for specific needs.</p>
</details>

<details>
  <summary><strong>Q13: What are the two resource types available for Azure AI Speech?</strong></summary>
  <p>• Speech resource – for standalone use and separate billing. <br>
  • Azure AI services resource – for integration with other Azure AI services and unified billing.</p>
</details>

<details>
  <summary><strong>Q14: Which interfaces and tools can be used to access Azure AI Speech?</strong></summary>
  <p>• Language Studio <br>
  • Command Line Interface (CLI) <br>
  • REST APIs <br>
  • SDKs</p>
</details>

<details>
  <summary><strong>Q15: What must your application do to use real-time transcription?</strong></summary>
  <p>It must capture audio input, stream it to the service, and receive the transcribed text in return.</p>
</details>

## 📘 Fundamentals of language translation
<details>
  <summary><strong>Q1: What is the main limitation of literal translation in AI systems?</strong></summary>
  <p>Literal translation may not account for context, grammar, or cultural idioms, which can lead to inaccurate or awkward translations.</p>
</details>

<details>
  <summary><strong>Q2: How does semantic translation improve upon literal translation?</strong></summary>
  <p>Semantic translation considers the meaning and context of words, enabling more accurate and natural translations.</p>
</details>

<details>
  <summary><strong>Q3: What model does Azure AI Translator use for its translations?</strong></summary>
  <p>It uses a Neural Machine Translation (NMT) model, which analyzes semantic context for more accurate translations.</p>
</details>

<details>
  <summary><strong>Q4: How many languages does Azure AI Translator support for text-to-text translation?</strong></summary>
  <p>Over 130 languages.</p>
</details>

<details>
  <summary><strong>Q5: What are ISO 639-1 language codes used for in Azure AI Translator?</strong></summary>
  <p>They are used to specify source and target languages, such as en for English or fr for French.</p>
</details>

<details>
  <summary><strong>Q6: What is custom translation in Azure AI Translator?</strong></summary>
  <p>It's a capability that allows developers to build customized NMT models tailored to their specific domain or vocabulary needs.</p>
</details>

<details>
  <summary><strong>Q7: What is profanity filtering in Azure AI Translator?</strong></summary>
  <p>It's an optional setting that lets you control how profane language is handled—either by filtering it out or marking it.</p>
</details>

<details>
  <summary><strong>Q8: What is selective translation in Azure AI Translator?</strong></summary>
  <p>It allows you to mark sections of text (like brand names or code) that should not be translated.</p>
</details>

<details>
  <summary><strong>Q9: What capabilities does Azure AI Speech support?</strong></summary>
  <p>It supports speech to text, text to speech, and speech translation.</p>
</details>

<details>
  <summary><strong>Q10: How is speech-to-text translation typically handled?</strong></summary>
  <p>The system transcribes spoken audio to text and can then translate the text to another language.</p>
</details>

<details>
  <summary><strong>Q11: How many languages does Azure AI Speech support for speech translation?</strong></summary>
  <p>Over 90 spoken languages.</p>
</details>

<details>
  <summary><strong>Q12: What format must the source language be specified in for Azure AI Speech translation?</strong></summary>
  <p>It must be in the extended language and culture code format, such as es-US for American Spanish.</p>
</details>

<details>
  <summary><strong>Q13: What is the difference between Azure AI Translator and Azure AI Speech?</strong></summary>
  <p>Azure AI Translator handles text-to-text translation, while Azure AI Speech enables speech-related capabilities, including translation and synthesis.</p>
</details>

<details>
  <summary><strong>Q14: What Azure resources can you create to use Translator or Speech services?</strong></summary>
  <p>You can create a Translator resource, a Speech resource, or a consolidated Azure AI services resource.</p>
</details>

<details>
  <summary><strong>Q15: What are the advantages of using an Azure AI services resource?</strong></summary>
  <p>It consolidates billing and access across multiple AI services and allows you to use a single endpoint and authentication key.</p>
</details>

## 📘 Computer Vision
<details>
  <summary><strong>Q1: What is the main difference between image processing and computer vision?</strong></summary>
  <p>Image processing applies filters for effects and visual manipulation, while computer vision focuses on extracting meaning or actionable insights from images using machine learning models.</p>
</details>

<details>
  <summary><strong>Q2: What type of machine learning model is commonly used in computer vision?</strong></summary>
  <p>Convolutional Neural Networks (CNNs) are commonly used due to their effectiveness in feature extraction and image classification.</p>
</details>

<details>
  <summary><strong>Q3: What is the role of filters in a CNN?</strong></summary>
  <p>Filters extract numeric feature maps from images that help in identifying key patterns, which are used for label prediction in classification tasks.</p>
</details>

<details>
  <summary><strong>Q4: How are CNN filter weights initialized and optimized?</strong></summary>
  <p>Filter weights are initialized with random values and adjusted during training to improve accuracy by comparing predicted outputs to known labels.</p>
</details>

<details>
  <summary><strong>Q5: What are transformers and how are they used in AI?</strong></summary>
  <p>Transformers are a neural network architecture primarily used in NLP that generate embeddings of tokens and can capture semantic relationships in language.</p>
</details>

<details>
  <summary><strong>Q6: How do multi-modal models extend the capabilities of transformers?</strong></summary>
  <p>Multi-modal models combine image encoders and language encoders to learn from both image data and textual captions, enabling tasks like classification, tagging, and captioning.</p>
</details>

<details>
  <summary><strong>Q7: What is Microsoft Florence?</strong></summary>
  <p>Florence is a multi-modal foundation model trained with large volumes of captioned images, capable of supporting image classification, object detection, captioning, and tagging.</p>
</details>

<details>
  <summary><strong>Q8: What is Azure AI Vision used for?</strong></summary>
  <p>Azure AI Vision is used to analyze images with capabilities like OCR, object detection, image captioning, and tagging, based on prebuilt or custom computer vision models.</p>
</details>

<details>
  <summary><strong>Q9: What are the two types of Azure resources you can create for using Azure AI Vision?</strong></summary>
  <p>• Azure AI Vision (dedicated to vision services)<br>• Azure AI Services (includes AI Vision and other services like Language and Translator)</p>
</details>

<details>
  <summary><strong>Q10: What are some image analysis capabilities of Azure AI Vision?</strong></summary>
  <p>OCR, image captioning, object detection, and tagging visual features.</p>
</details>

<details>
  <summary><strong>Q11: What are the benefits of using Azure AI Vision instead of training your own model?</strong></summary>
  <p>You save time and resources by leveraging prebuilt models, avoid the need for large datasets and compute power, and can still customize models with your own images.</p>
</details>

<details>
  <summary><strong>Q12: What is the benefit of foundation models like Florence in AI applications?</strong></summary>
  <p>They provide a general pre-trained model that can be adapted to specific tasks, reducing the effort required to build specialized models from scratch.</p>
</details>
<details>

  ## 📘 Fundamentals of Facial Recognition
  
  <summary><strong>Q1: What is face detection?</strong></summary>
  <p>Face detection involves identifying regions of an image that contain human faces, typically returning bounding box coordinates around each face.</p>
</details>

<details>
  <summary><strong>Q2: How does face analysis differ from face detection?</strong></summary>
  <p>While face detection identifies face locations, face analysis examines facial features such as eyes, nose, and mouth to provide deeper insights.</p>
</details>

<details>
  <summary><strong>Q3: What is facial recognition?</strong></summary>
  <p>Facial recognition is the process of identifying known individuals based on their facial features using trained machine learning models.</p>
</details>

<details>
  <summary><strong>Q4: Name two common applications of facial recognition.</strong></summary>
  <p>Security (e.g., unlocking phones) and social media (e.g., auto-tagging friends).</p>
</details>

<details>
  <summary><strong>Q5: Which Azure AI service offers the most extensive facial analysis capabilities?</strong></summary>
  <p>Azure AI Face service.</p>
</details>

<details>
  <summary><strong>Q6: What is the difference between Azure AI Face and Azure AI Vision for face-related tasks?</strong></summary>
  <p>Azure AI Face provides advanced facial recognition and analysis features, whereas Azure AI Vision focuses on basic face detection.</p>
</details>

<details>
  <summary><strong>Q7: What image formats are supported by Azure AI Face service?</strong></summary>
  <p>JPEG, PNG, GIF, and BMP.</p>
</details>

<details>
  <summary><strong>Q8: What is the maximum file size supported by Azure AI Face service?</strong></summary>
  <p>6 MB.</p>
</details>

<details>
  <summary><strong>Q9: What is the supported face size range for detection in Azure AI Face service?</strong></summary>
  <p>From 36 x 36 pixels up to 4096 x 4096 pixels.</p>
</details>

<details>
  <summary><strong>Q10: What facial attributes can Azure AI Face return?</strong></summary>
  <p>Accessories, blur, exposure, glasses, head pose, mask, noise, occlusion, and quality for recognition.</p>
</details>

<details>
  <summary><strong>Q11: What does the ‘quality for recognition’ attribute represent?</strong></summary>
  <p>It indicates whether an image is of sufficient quality for facial recognition (rated high, medium, or low).</p>
</details>

<details>
  <summary><strong>Q12: What is occlusion in face analysis?</strong></summary>
  <p>It refers to objects partially blocking parts of the face, such as a hand or sunglasses.</p>
</details>

<details>
  <summary><strong>Q13: What does the ‘blur’ attribute signify in Azure AI Face?</strong></summary>
  <p>It measures how blurred the face is, which affects its suitability for analysis.</p>
</details>

<details>
  <summary><strong>Q14: How does head pose analysis work?</strong></summary>
  <p>It provides the orientation of the face in 3D space, indicating where the person is looking.</p>
</details>

<details>
  <summary><strong>Q15: Which Azure service helps analyze faces in videos?</strong></summary>
  <p>Azure AI Video Indexer.</p>
</details>

<details>
  <summary><strong>Q16: Which service can perform liveness detection?</strong></summary>
  <p>Azure AI Face service (requires Limited Access).</p>
</details>

<details>
  <summary><strong>Q17: What is required to access advanced features like face identification or liveness detection in Azure AI Face?</strong></summary>
  <p>Submitting an intake form to request Limited Access.</p>
</details>

<details>
  <summary><strong>Q18: What are some applications of facial analysis in real-world scenarios?</strong></summary>
  <p>Driver monitoring, advertising demographic analysis, identity validation, and missing persons identification.</p>
</details>

<details>
  <summary><strong>Q19: What can impair face detection accuracy?</strong></summary>
  <p>Extreme angles, poor lighting, and occlusions.</p>
</details>

<details>
  <summary><strong>Q20: What two resource types can be created to use Azure AI Face?</strong></summary>
  <p>• Face (dedicated resource)<br>• Azure AI services (shared resource for multiple AI services)</p>
</details>

<details>
  <summary><strong>Q21: What attribute helps detect whether a person is wearing a mask or not?</strong></summary>
  <p>The mask attribute.</p>
</details>

<details>
  <summary><strong>Q22: How does facial recognition differ from facial detection?</strong></summary>
  <p>Recognition identifies who the person is; detection only finds that a face is present.</p>
</details>

## 📘 Optical Character Recognition
<details>
  <summary><strong>Q1: What is the purpose of Optical Character Recognition (OCR)?</strong></summary>
  <p>OCR enables AI systems to read and extract text from images such as scanned documents or photographs.</p>
</details>

<details>
  <summary><strong>Q2: Which two AI domains intersect in OCR tasks?</strong></summary>
  <p>Computer Vision and Natural Language Processing.</p>
</details>

<details>
  <summary><strong>Q3: What is the basic approach OCR uses to recognize text in images?</strong></summary>
  <p>It uses machine learning models trained to recognize shapes as letters, numbers, punctuation, or other text elements.</p>
</details>

<details>
  <summary><strong>Q4: What common tasks does OCR help automate?</strong></summary>
  <p>Note taking, scanning checks, digitizing records, and extracting data from documents.</p>
</details>

<details>
  <summary><strong>Q5: Which Azure service provides OCR capabilities?</strong></summary>
  <p>Azure AI Vision.</p>
</details>

<details>
  <summary><strong>Q6: What is the name of the OCR engine in Azure AI Vision?</strong></summary>
  <p>The Read API, also called the Read OCR engine.</p>
</details>

<details>
  <summary><strong>Q7: What file formats are supported by Azure AI Vision's Read API?</strong></summary>
  <p>Images (e.g., JPEG, PNG), PDF, and TIFF files.</p>
</details>

<details>
  <summary><strong>Q8: What kind of images is the Read OCR engine optimized for?</strong></summary>
  <p>General, non-document images that contain significant text or visual noise.</p>
</details>

<details>
  <summary><strong>Q9: What are the three hierarchical levels of OCR output from the Read API?</strong></summary>
  <p>Pages → Lines → Words.</p>
</details>

<details>
  <summary><strong>Q10: What information does each word object in the OCR output include?</strong></summary>
  <p>The bounding box coordinates and the detected word text.</p>
</details>

<details>
  <summary><strong>Q11: What do bounding boxes in OCR indicate?</strong></summary>
  <p>The position and region of detected text elements within an image.</p>
</details>

<details>
  <summary><strong>Q12: What resource types can you create to use Azure AI Vision OCR?</strong></summary>
  <p>Azure AI Vision or Azure AI Services.</p>
</details>

<details>
  <summary><strong>Q13: When should you choose an Azure AI Vision resource instead of Azure AI Services?</strong></summary>
  <p>When you only plan to use Vision services and want to track their usage separately.</p>
</details>

<details>
  <summary><strong>Q14: What happens when you use Azure AI Vision to process images in Vision Studio?</strong></summary>
  <p>Your resource begins to incur usage costs.</p>
</details>

<details>
  <summary><strong>Q15: What is the purpose of Azure Vision Studio for OCR?</strong></summary>
  <p>To try out and evaluate OCR functionality using sample or user-provided images.</p>
</details>

<details>
  <summary><strong>Q16: What options are available to integrate OCR programmatically?</strong></summary>
  <p>REST API and SDKs in Python, C#, and JavaScript.</p>
</details>

<details>
  <summary><strong>Q17: What is returned by the Vision Studio after analyzing an image?</strong></summary>
  <p>JSON results with bounding box locations and detected text.</p>
</details>

<details>
  <summary><strong>Q18: How did OCR originally gain traction in real-world usage?</strong></summary>
  <p>Through postal services automating the sorting of mail using postal codes.</p>
</details>

<details>
  <summary><strong>Q19: How does OCR improve operational efficiency?</strong></summary>
  <p>By removing the need for manual data entry and accelerating text processing.</p>
</details>

<details>
  <summary><strong>Q20: Why is OCR useful in digitizing historical documents?</strong></summary>
  <p>It can convert printed or handwritten text into machine-readable formats for preservation and analysis.</p>
</details>

## 📘 Generative AI workloads on Azure

<details>
  <summary><strong>Q1: What is the primary use of Azure OpenAI service?</strong></summary>
  <p>To access generative AI models like GPT and DALL-E for text and image generation within a scalable and secure cloud environment.</p>
</details>

<details>
  <summary><strong>Q2: Which Azure AI service enables you to extract text, tags, and descriptions from images?</strong></summary>
  <p>Azure AI Vision.</p>
</details>

<details>
  <summary><strong>Q3: You want to convert spoken audio into text and also perform speech translation. Which service should you use?</strong></summary>
  <p>Azure AI Speech.</p>
</details>

<details>
  <summary><strong>Q4: What does the Azure AI Language service provide?</strong></summary>
  <p>Models for tasks like sentiment analysis, summarization, entity recognition, and building conversational agents.</p>
</details>

<details>
  <summary><strong>Q5: Which Azure AI service should be used to detect and flag offensive or harmful content in text and images?</strong></summary>
  <p>Azure AI Content Safety.</p>
</details>

<details>
  <summary><strong>Q6: What does Azure AI Translator do?</strong></summary>
  <p>Translates text between many different languages using advanced language models.</p>
</details>

<details>
  <summary><strong>Q7: Which AI service is designed to detect and recognize human faces in images?</strong></summary>
  <p>Azure AI Face.</p>
</details>

<details>
  <summary><strong>Q8: Can you use Azure AI Custom Vision to detect objects and classify images based on custom-trained models?</strong></summary>
  <p>Yes.</p>
</details>

<details>
  <summary><strong>Q9: What is the primary use of Azure AI Document Intelligence?</strong></summary>
  <p>To extract structured data from complex documents like forms, invoices, and receipts.</p>
</details>

<details>
  <summary><strong>Q10: What Azure service would you use to analyze images, videos, and audio streams simultaneously?</strong></summary>
  <p>Azure AI Content Understanding.</p>
</details>

<details>
  <summary><strong>Q11: What is the purpose of Azure AI Search?</strong></summary>
  <p>To create searchable indexes using AI-extracted information from documents for retrieval and grounding in applications like RAG systems.</p>
</details>

<details>
  <summary><strong>Q12: What are two types of Azure AI resources you can create?</strong></summary>
  <p>Standalone Azure AI services and multi-service Azure AI services resources.</p>
</details>

<details>
  <summary><strong>Q13: Which Azure AI resource type should you use if you want to use multiple services and simplify cost and access management?</strong></summary>
  <p>Multi-service Azure AI services resource.</p>
</details>

<details>
  <summary><strong>Q14: What is one advantage of using standalone Azure AI services?</strong></summary>
  <p>They often offer free-tier SKUs for evaluation and development.</p>
</details>

<details>
  <summary><strong>Q15: What should you check before provisioning a specific Azure AI service in a region?</strong></summary>
  <p>Regional availability and quota restrictions.</p>
</details>

<details>
  <summary><strong>Q16: What tool can you use to estimate the cost of using Azure AI services?</strong></summary>
  <p>Azure Pricing Calculator.</p>
</details>

<details>
  <summary><strong>Q17: If you're building a document processing solution to extract line items from scanned invoices, which Azure service should you use?</strong></summary>
  <p>Azure AI Document Intelligence.</p>
</details>

<details>
  <summary><strong>Q18: What is one benefit of provisioning Azure AI services as part of an Azure Foundry hub?</strong></summary>
  <p>Centralized access control and cost management across projects.</p>
</details>

<details>
  <summary><strong>Q19: What kind of interface does Custom Vision provide for uploading training images and deploying models?</strong></summary>
  <p>A web-based visual interface.</p>
</details>

<details>
  <summary><strong>Q20: How are Azure AI services typically accessed by applications?</strong></summary>
  <p>Through service-specific APIs and SDKs using endpoints and authorization keys.</p>
</details>

## 📘 AI Agents
<details>
  <summary><strong>Q1: What is an AI agent in the context of Azure AI solutions?</strong></summary>
  <p>An AI agent is a smart software service that combines generative AI models with contextual data and automation capabilities to perform tasks based on user input and environmental cues.</p>
</details>

<details>
  <summary><strong>Q2: What is a practical example of an AI agent in a business setting?</strong></summary>
  <p>An AI agent that helps employees manage expense claims, answers policy-related questions, and automatically submits or routes expense reports.</p>
</details>

<details>
  <summary><strong>Q3: What are two core capabilities AI agents must possess?</strong></summary>
  <p>Access to generative AI models and the ability to automate tasks based on contextual and user input.</p>
</details>

<details>
  <summary><strong>Q4: How do AI agents enhance productivity in organizations?</strong></summary>
  <p>By automating repetitive tasks, orchestrating workflows, and providing intelligent assistance to users.</p>
</details>

<details>
  <summary><strong>Q5: What is Azure AI Agent Service?</strong></summary>
  <p>A managed service in Azure designed to build, manage, and deploy AI agents with enterprise-grade features and integration into Azure AI Foundry.</p>
</details>

<details>
  <summary><strong>Q6: What framework is Azure AI Agent Service based on?</strong></summary>
  <p>The OpenAI Assistants API, with added flexibility, model support, and enterprise features.</p>
</details>

<details>
  <summary><strong>Q7: How does Azure AI Agent Service differ from OpenAI Assistants API?</strong></summary>
  <p>Azure AI Agent Service supports more models, enhanced data integration, and enterprise security features, whereas OpenAI Assistants API only works with OpenAI models.</p>
</details>

<details>
  <summary><strong>Q8: What is Semantic Kernel used for?</strong></summary>
  <p>It is an open-source development kit used for building AI agents and orchestrating multi-agent solutions.</p>
</details>

<details>
  <summary><strong>Q9: What is the Semantic Kernel Agent Framework?</strong></summary>
  <p>A specialized framework optimized for building and managing AI agents using Semantic Kernel.</p>
</details>

<details>
  <summary><strong>Q10: What is AutoGen in the context of AI agent development?</strong></summary>
  <p>An open-source framework useful for rapid prototyping and research into agent-based applications.</p>
</details>

<details>
  <summary><strong>Q11: What is the Microsoft 365 Agents SDK used for?</strong></summary>
  <p>To build and self-host AI agents that can be delivered across channels like Microsoft Teams, Slack, or Messenger.</p>
</details>

<details>
  <summary><strong>Q12: Does Microsoft 365 Agents SDK restrict agents to Microsoft 365 environments only?</strong></summary>
  <p>No, agents can be deployed in multiple channels beyond Microsoft 365.</p>
</details>

<details>
  <summary><strong>Q13: What is Microsoft Copilot Studio?</strong></summary>
  <p>A low-code platform for building AI agents, allowing non-developers to create and deploy agents within Microsoft 365 and external channels.</p>
</details>

<details>
  <summary><strong>Q14: Who is the primary target audience for Copilot Studio agent builder?</strong></summary>
  <p>Business users or “citizen developers” with little or no programming experience.</p>
</details>

<details>
  <summary><strong>Q15: What type of interface does Copilot Studio provide for agent creation?</strong></summary>
  <p>A visual drag-and-drop interface and a declarative language-based configuration option.</p>
</details>

<details>
  <summary><strong>Q16: Which tool is ideal for non-technical users to create simple task automation agents?</strong></summary>
  <p>Copilot Studio agent builder in Microsoft 365 Copilot.</p>
</details>

<details>
  <summary><strong>Q17: Which tool should be used for building low-code solutions that integrate with Microsoft 365?</strong></summary>
  <p>Microsoft Copilot Studio.</p>
</details>

<details>
  <summary><strong>Q18: What is the best choice for professional developers creating complex agents integrated with Azure backend services?</strong></summary>
  <p>Azure AI Agent Service.</p>
</details>

<details>
  <summary><strong>Q19: When is Semantic Kernel recommended in AI agent development?</strong></summary>
  <p>When building multi-agent solutions or orchestrating interactions between multiple agents.</p>
</details>

<details>
  <summary><strong>Q20: How does Azure AI Agent Service integrate with Azure Foundry?</strong></summary>
  <p>It centralizes access control, integrates with Azure AI services, and supports scalable development of AI agents using enterprise governance.</p>
</details>


