### Problem (unicode 1): Understanding Unicode
1. '\x00'
2. The `__repr__()` function returns a quoted string literal of itself, that is "'\\x00'"
3. The print function will omit any non-printable character and outside quotes.

### Problem (unicode 2): Unicode Encodings
1. Because the vocabulary will be much larger and the encoding will take up more space of 2 bytes and 4 bytes instead of 1.
2. "nihao"(Chinese), when a byte does not correspond to one unicode character
3. b"\xff\xff", when a byte is invalid for utf-8

### Problem (transformer_accounting):
1. vocab_size, context_length, num_layers, d_model, num_heads, d_ff
The total parameter will be vocab_size * context_length + num_layers * (d_model + d_model * d_model * 4 + d_ff * d_model * 3) + vocab_size * d_model
2. According to the architecture figure:
embedding layer: (batch_size, seq_len, num_embedding) * (num_embedding, embedding_dim) -> (batch_size, seq_len, embedding_dim), total batch_size * seq_len * num_embedding * embedding_dim flops
transformer layers: in each transformer block, we have 2 norm layers, 1 multi-head attention with rope and 1 ffn. For each norm layer, we have batch_size * d_model * 4 flops. For each ffn layer, we have batch_size * (d_model * d_ffn + d_ffn * d_model + d_ffn * d_model + d_ffn + d_model). For multi-head attention block with casual masking, there are batch_size * d_model * d_model * seq_len * 3 + batch_size * (d_model // num_heads) * (d_model // num_heads) * d_model * 2. And for softmax, batch_size * num_heads * (d_model // num_heads) * (d_model // num_heads) * 3. For the two ropes, each of which has batch_size * seq_num * num_heads * 2 * (d_model // num_heads) * (d_model // num_heads) * d_model. So put it together, we have (batch_size * d_model * 4 + batch_size * (d_model * d_ffn) * 4 + batch_size * d_model * d_model * seq_len * 3 + batch_size * (d_model // num_heads) * (d_model // num_heads) * d_model * 2 + batch_size * num_heads * (d_model // num_heads) * (d_model // num_heads) * 3 + batch_size * seq_num * num_heads * 2 * (d_model // num_heads) * (d_model // num_heads) * d_model) * num_layers.
For the final rms_norm and linear layer, we have vocab_size * d_model * batch_size + batch_size * 4 * d_model.
3. transformer blocks, especially attention mechanisms.
4. d_model -> cubic increase. num_layers: linear increase. num_heads: reduce.

### Problem (adamwAccounting)
1. Based on analysis on the last section, total memory = param_num * 3 (m, v, param itself). However we can repeatedly use some of the memory, so we will just calculate the maximum memory usage of layer and multiply it by 3.
3. 8 * param_num
