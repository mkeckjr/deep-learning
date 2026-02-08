import tiktoken

from keck_llm.data import GPTDataset, gpt_in_memory_dataloader

def test_gpt_dataset_gpt2(small_text_path):
    with open(small_text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    loader = gpt_in_memory_dataloader(text)
    for i, (context, target) in enumerate(loader):
        print(f'Batch {i}')
        
