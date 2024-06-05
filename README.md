# Re-Tuning: Overcoming the Compositionality Limits of Large Language Models with Recursive Tuning
The source code repo for ACL 2024 paper [Re-Tuning: Overcoming the Compositionality Limits of Large Language Models with Recursive Tuning](https://arxiv.org/), by Eric Pasewark, Kyle Montgomery, Kefei Duan, Dawn Song and Chenguang Wang.

We train a language model to write prompts to itself as part of its problem-solving process. The readme file is organized as follows: first a walkthrough of running data geneartion, training, and evaluation on addition. Then more details on how the code works are provided.
# Training and Evaluation Process
1. For each problem, generate fine-tuning data that includes prompts to the model.
1. Fine-tune on this data.
1. Evaluate accuracy on the problem. All problems looked at so far have simple ways to evaluate accuracy.

# Full Walkthrough of Running Training and Evaluation for Addition
Here is the process to reproduce our results.
## Data Generation
Simply run all the cells in one of the data generation notebooks. This will make a json file with the inputs and outputs we want the model to learn from. Put this file in the same directory as the train.py file.

## Fine-Tuning
Any GPU with 40GB or more memory should be sufficient for fine-tuning Llama 7b with LoRA. Open a terminal in the directory with the code and json file. Then run the command

    python -u train.py model=llama7b datasets=[YOUR_JSON_FILE_HERE.json] exp_name=my_new_experiment gradient_accumulation_steps=32 batch_size=128 eval_batch_size=4 n_eval_examples=16

For the datasets parameter, put the name of the json file inside brackets. You may need to increase the gradient_accumulation_steps if you get an out of memory error.

If it is now running, you will see it download the model, load the data, then start printing training loss. About every 8000 training examples it will save the LORA weights inside the cache folder (you can change this with the eval_every variable in the config file). 

## Evaluation
Choose the evaluation notebook from the "evaluation" folder that matches the data the model saw in training. For example, if the generated data was for Re-Tuning addition, use the Re-Tuning addition evaluation notebook. Change the 'lora_dir' variable in the notebook to the directory the LORA weights are in. Then just run the cells. This will compute accuracies and make a plot. You can also adjust which problem lengths you want to evaluate on and how many samples per problem length in the notebook.

# Code Details
## Data Generation and Finetuning
The Re-Tuning data is similar to scratchpad-finetuning data in that it should contain intermediate steps to solve the problem. However, the data should also consist of prompts to the model enclosed between "Call: " and "\n". For example, a prompt and completion in the addition problem looks like this:

    {'input': '73199 + 21439\nSolution: ', 'output': 'Call: 3199 + 1439\n'}

Here the model is prompted to add 2 numbers and should generate a prompt to add 2 smaller numbers as part of its solution.
The data format is like the example above, where the prompt and response pairs are in dictionaries with 'input' and 'output' keys. This is stored as json files.
Notebooks to create this data are in the data_generation folder.

If the model does generate a prompt, during inference it should recieve the answer to that prompt. After recieving the answer, the model should continue generation, which may include other prompts, and eventually the final answer. We break up this process into multiple examples for fine-tuning. 
In the example to add 73199 + 21439 from before, one prompt-response pair is

    {'input': '73199 + 21439\nSolution: ', 'output': 'Call: 3199 + 1439\n'}

In this pair we teach the model to generate the prompt for the subproblem. Then we have a separate training example where the model recieves the output from the call and then outputs the final answer:

    {'input': '73199 + 21439\nSolution: Call: 3199 + 1439\nReturn: Carry 0, Output 4638\nAnswer: ','output': 'Carry 0, Output 94638'}.

The main reason for doing this is that we do not want the model to try to predict the output from the prompts it generates, which may waste model capacity. The code for fine-tuning does not compute training loss on the prompt text. So when we break up the examples like this, the model does not try to predict the return from the call. 

We could also achieve this by sending the entire context in for fine-tuning as usual, and then masking both the prompt and the return. There are 2 reasons to favor the current implementation though. Masking multiple sections of input is trickier to implement compared to masking just the prompts. Additionally, when the examples are split the model automatically learns to generate the eos token after calls. So during inference the model will stop generation after creating a call. Otherwise we would need to constantly check the model output during generation to see if there are any recursive calls.

The training is done with LORA to use less memory. For this, the GPU should have around 40GB or more of memory for using LoRA with Llama 7b. 

The finetuning code is a heavily modified version of the code from https://github.com/eric-mitchell/direct-preference-optimization.

## Evaluation
The evaluation is done in notebooks in the "evaluation" folder. The evaluation is fairly straightforward. A prompt is generated and sent to the model. The prompt follows the same prompt template as the model saw in fine-tuning. Then we compare the final output to the true answer and record that for the accuracy. Since the model output is the string it generated, we need to extract the final answer from the string by some method. This varies by the problem. 
- For addition the end of the output string should have a carry and output. We extract these by splitting the string on ' '. If the carry is 1, we prepend that to the output, then compare that string to the string of the true number.
- For dynamic programming, we split the output string on 'Return: '. The text after 'Return: ' should be the final answer. We compare this string to the string of the true answer.
- For parity, we split the output string on ' ' and take the last element of this list. Then we compare this to the string of the true answer.


### Citation
```bibtex
@InProceedings{pasewark2024retuning,
  title={Re-Tuning: Overcoming the Compositionality Limits of Large Language Models with Recursive Tuning},
  author={Pasewark, Eric and Montgomery, Kyle and Duan, Kefei and Song, Dawn and Wang, Chenguang},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  publisher={Association for Computational Linguistics},
  year={2024}
}
```
