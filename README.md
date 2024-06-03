# Re-Tuning: Overcoming the Compositionality Limits of Large Language Models with Recursive Tuning
The source code repo for ACL 2024 paper [Re-Tuning: Overcoming the Compositionality Limits of Large Language Models with Recursive Tuning](https://arxiv.org/).
We train a language model to write prompts to itself as part of its problem-solving process. The readme file is organized as follows: first a walkthrough of running data geneartion, training, and evaluation on addition. Then more details on how the code works are provided.
# Training and Evaluation Process
1. For each problem, generate fine-tuning data that includes prompts to the model.
1. Fine-tune on this data.
1. Evaluate accuracy on the problem. All problems looked at so far have simple ways to evaluate accuracy.

# Full Walkthrough of Running Training and Evaluation for Addition
Here I will go through the entire process to get results on addition. The process for other problems is very similar.
## Data Generation
Simply run all the cells in the 'Create Addition Recursive Data' notebook. This will make a json file with the inputs and outputs we want the model to learn from.

## Fine-Tuning
Any GPU with 40GB or more memory should be sufficient for fine-tuning. Open a terminal in the directory with the code and json file. Then run the command

    python -u train.py model=llama7b datasets=[recursive_add_split.json] loss=sft exp_name=testing gradient_accumulation_steps=32 batch_size=128 eval_batch_size=4 n_eval_examples=16 sample_during_eval=false

For the datasets parameter, put the name of the json file inside brackets. For the resampling parameter put either addition, dp, or parity depending on which problem. You may need to increase the gradient_accumulation_steps if you get an out of memory error.

If it is now running, you will see it download the model, load the data, then start printing training loss. About every 8000 training examples it will save the LORA weights inside the cache folder (you can change this with the eval_every variable in the config file). 

## Evaluation
The notebook to use for addition is 'Evaluate addition llama 7b-recursive'. Change the 'lora_dir' variable to the directory the LORA weights are in. Depending on your setup you may want to comment out the 'os.environ["CUDA_VISIBLE_DEVICES"]="1"' line. Then just run the first 7 or so cells. This will compute accuracies and make a plot.

# Code Details
## Fine-tuning data generation
The data should be similar to scratchpad-finetuning data in that it should contain intermediate steps to solve the problem. However, the data should also consist of prompts to the model enclosed between "Call: " and "\n". For example, a prompt and completion in the addition problem looks like this:

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
## Fine-tuning process
The code for fine-tuning is run on command-line. Example commands look like this:

    python -u train.py model=llama7b datasets=[dp_recursive] loss=sft exp_name=testing gradient_accumulation_steps=32 batch_size=128 eval_batch_size=4 n_eval_examples=16 sample_during_eval=false

For arithmetic we can do:

    python -u train.py model=llama7b datasets=[arithmetic_recursive] loss=sft exp_name=testing gradient_accumulation_steps=32 batch_size=128 eval_batch_size=4 n_eval_examples=16 sample_during_eval=false

The training is done with LORA to use less memory. For this, the GPU should have around 40GB or more of memory. 

## Evaluation
I have done the evaluation in Jupyter notebooks. The evaluation I have done is fairly straightforward. I prompt the model on the problem, following the same prompt template as the model saw in fine-tuning. Then we compare the final output to the true answer and record that for the accuracy. Since the model output is the string it generated, we need to extract the final answer from the string by some method. This varies by the problem. 
- For addition the end of the output string should have a carry and output. We extract these by splitting the string on ' '. If the carry is 1, we prepend that to the output, then compare that string to the string of the true number.
- For dynamic programming, we split the output string on 'Return: '. The text after 'Return: ' should be the final answer. We compare this string to the string of the true answer.
- For parity, we split the output string on ' ' and take the last element of this list. Then we compare this to the string of the true answer.

To get the final results in the graphs from my write-up, I sample 20-40 examples per data point, depending on the problem. So for addition for example, to compute accuracy on n digits I randomly generate 20 n-digit numbers and run the model on them.

The model requires less memory for evaluation and these can be done on 24GB gpu, and maybe even 16GB. 

# Code Descriptions
For fine-tuning, the important python files are preference_datasets and trainers. trainers contains the code to compute the training loss and update the model. I don't think we should have to make modifications to this file very much. preference_datasets contains the code to load and tokenize the data. This is where we will need to add or modify code to train the model on different datasets.

- For evaluation, the code is contained in notebooks in the eval folder which are titled by which data they evaluate.
- For data generation, the notebooks are in data_generation and titled by what data they generate.

Note that for some of these the paths will need to be modified for whataver machine you run on. So for example in evaluation the 'lora_dir' variable should point to the directory with the LORA weights.

Most evaluation notebooks contain the evaluation of multiple models. Usually these evaluations take 2-4 cells. The first cell usually has a comment with the model name. This will usually be a short description and then the number of training examples seen. For example, a comment could be

    # llama_dp_itertools_full_resample5_95232

Here the model string has the problem (dynamic programming), some specific training information (generated all training examples up to length 5, and did resampling scheme), and shows that the model saw 95232 examples during training.

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