import os
import time
import logging
import keras
import openai
import dotenv
import bentoml
import numpy as np
import pandas as pd
from keras import layers




# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler and set the log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it to the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DATA_FILES_ROOT = "../dataset/data/"
CONVERSATIONS_CSV_PATH = os.path.join(DATA_FILES_ROOT, "conversations.csv")
EMBEDDINGS_CACHE_PATH = os.path.join(DATA_FILES_ROOT, "vectors.npy")

client = openai.OpenAI()


def load_data():
    dotenv.load_dotenv()

    # Load the data
    raw_df = pd.read_csv(CONVERSATIONS_CSV_PATH)[["message", "message_type"]]
    raw_df.rename(columns={
        "message": "text",
        "message_type": "label"
    }, inplace=True)
    raw_df.head()

    return raw_df


def clean_data(raw_df):
    allowed_labels = {'question', 'answer', 'comment'}
    raw_df['label'] = raw_df['label'].str.lower()
    raw_df = raw_df[raw_df['label'].isin(allowed_labels)]
    raw_df['label'].value_counts()

    return raw_df


def embed_data(raw_df):
    vectors = []

    number_of_samples = len(raw_df)

    for idx, text in enumerate(raw_df['text']):
        print(f"{idx}/{number_of_samples} - {text}")
        try:
            response = client.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL,
            )
            vector = response.data[0].embedding
            vectors.append(vector)
        except openai.APIError:
            logger.error(f"Failed to embed text: {text}")
            time.sleep(5)
            continue
        except openai.error.ServiceUnavailableError:
            logger.error("OpenAI Service is unavailable")
            break

    vectors_arr = np.asarray(vectors, dtype=np.float64)
    np.save(EMBEDDINGS_CACHE_PATH, vectors_arr)


def train_test_split(raw_df):
    def label_encoder(label):
        return {
            "question": 0,
            "answer": 1,
            "comment": 2
        }[label]

    vectors_arr = np.load(EMBEDDINGS_CACHE_PATH)
    df_full = raw_df.copy()
    df_full['label'] = raw_df['label']
    df_full['label_id'] = df_full['label'].apply(label_encoder)
    df_full['vector'] = list(vectors_arr)

    train_size = 0.8

    df_train = df_full.sample(frac=train_size, random_state=42)
    df_test = df_full.drop(df_train.index).reset_index(drop=True)

    return df_train, df_test


def create_model(df_train):
    embedding_size = len(df_train['vector'].iloc[0])
    num_classes = len(df_train['label'].unique())
    input_layer = keras.Input((embedding_size,))
    hidden_layer = layers.Dense(embedding_size, activation='relu')(input_layer)
    output_layer = layers.Dense(num_classes, activation='softmax')(hidden_layer)
    classifier = keras.Model(
        inputs=[
            input_layer
        ],
        outputs=output_layer,
    )
    classifier.summary()

    classifier.compile(
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return classifier


def train_model(classifier, df_train, df_test):
    NUM_EPOCHS = 40
    BATCH_SIZE = 32

    # Split the x and y components of the train and validation subsets.
    y_train = df_train['label_id']
    x_train = np.stack(df_train['vector'])
    y_test = df_test['label_id']
    x_test = np.stack(df_test['vector'])

    # Train the model for the desired number of epochs.
    callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

    class_counts = df_train['label_id'].value_counts()
    total_count = class_counts.sum()
    class_weight = {
        label: round(total_count / count, 4)
        for label, count in class_counts.items()
    }
    print("Class Weights: ", class_weight)

    history = classifier.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        callbacks=[callback],
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        # class_weight=class_weight,
    )

    return history


def export_model(classifier):
    bentoml.tensorflow.save_model("qa_classifier", classifier)


def main():
    raw_df = load_data()
    raw_df = clean_data(raw_df)
    embed_data(raw_df)
    df_train, df_test = train_test_split(raw_df)
    classifier = create_model(df_train)
    train_model(classifier, df_train, df_test)
    export_model(classifier)


if __name__ == "__main__":
    main()
