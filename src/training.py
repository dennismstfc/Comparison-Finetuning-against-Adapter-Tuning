from transformers import TrainingArguments

def get_training_args(output_dir, logging_dir):
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_strategy="epoch",
        logging_steps=100,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-6,
        save_strategy="epoch",
        save_steps=100,
        evaluation_strategy="epoch",
        eval_steps=100,
        load_best_model_at_end=True
    )