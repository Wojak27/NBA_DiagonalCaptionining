"""
<a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!--- @wandbcode{sweeps-video} -->
"""

"""
<img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

<!--- @wandbcode{sweeps-video} -->

<div><img /></div>

<img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>
"""

"""

# 🧹 Introduction to Hyperparameter Sweeps using W&B

Searching through high dimensional hyperparameter spaces to find the most performant model can get unwieldy very fast. Hyperparameter sweeps provide an organized and efficient way to conduct a battle royale of models and pick the most accurate model. They enable this by automatically searching through combinations of hyperparameter values (e.g. learning rate, batch size, number of hidden layers, optimizer type) to find the most optimal values.

In this tutorial we'll see how you can run sophisticated hyperparameter sweeps in 3 easy steps using Weights and Biases.

### Follow along with a [video tutorial](http://wandb.me/sweeps-video)!

![](https://i.imgur.com/WVKkMWw.png)

## Sweeps: An Overview

Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

1. **Define the sweep:** we do this by creating a dictionary or a [YAML file](https://docs.wandb.com/library/sweeps/configuration) that specifies the parameters to search through, the search strategy, the optimization metric et all.

2. **Initialize the sweep:** with one line of code we initialize the sweep and pass in the dictionary of sweep configurations:
`sweep_id = wandb.sweep(sweep_config)`

3. **Run the sweep agent:** also accomplished with one line of code, we call `wandb.agent()` and pass the `sweep_id` to run, along with a function that defines your model architecture and trains it:
`wandb.agent(sweep_id, function=train)`

And voila! That's all there is to running a hyperparameter sweep! In the notebook below, we'll walk through these 3 steps in more detail.


We highly encourage you to fork this notebook so you can tweak the parameters,
try out different models,
or try a Sweep with your own dataset!

## Resources
- [Sweeps docs →](https://docs.wandb.ai/sweeps)
- [Launching from the command line →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

"""

"""
# 🚀 Setup

Start out by installing the experiment tracking library and setting up your free W&B account:

1. Install with `!pip install`
2. `import` the library into Python
3. `.login()` so you can log metrics to your projects

If you've never used Weights & Biases before,
the call to `login` will give you a link to sign up for an account.
W&B is free to use for personal and academic projects!
"""


import wandb
from main_task_action_multifeat_multilevel_CLIP import main
from main_task_caption_CLIP import Args_Caption



wandb.login()

"""
# Step 1️⃣. Define the Sweep

Fundamentally, a Sweep combines a strategy for trying out a bunch of hyperparameter values with the code that evalutes them.
Whether that strategy is as simple as trying every option
or as complex as [BOHB](https://arxiv.org/abs/1807.01774),
Weights & Biases Sweeps have you covered.
You just need to _define your strategy_
in the form of a [configuration](https://docs.wandb.com/sweeps/configuration).

When you're setting up a Sweep in a notebook like this,
that config object is a nested dictionary.
When you run a Sweep via the command line,
the config object is a
[YAML file](https://docs.wandb.com/sweeps/quickstart#2-sweep-config).

Let's walk through the definition of a Sweep config together.
We'll do it slowly, so we get a chance to explain each component.
In a typical Sweep pipeline,
this step would be done in a single assignment.
"""

"""
### 👈 Pick a `method`
"""

"""
The first thing we need to define is the `method`
for choosing new parameter values.

We provide the following search `methods`:
*   **`grid` Search** – Iterate over every combination of hyperparameter values.
Very effective, but can be computationally costly.
*   **`random` Search** – Select each new combination at random according to provided `distribution`s. Surprisingly effective!
*   **`bayes`ian Search** – Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Works well for small numbers of continuous parameters but scales poorly.

We'll stick with `random`.
"""

sweep_config = {
    'method': 'grid'
    }

"""
For `bayes`ian Sweeps,
you also need to tell us a bit about your `metric`.
We need to know its `name`, so we can find it in the model outputs
and we need to know whether your `goal` is to `minimize` it
(e.g. if it's the squared error)
or to `maximize` it
(e.g. if it's the accuracy).
"""

metric = {
    'name': 'Bleu_4',
    'goal': 'maximize'
    }

sweep_config['metric'] = metric

"""
If you're not running a `bayes`ian Sweep, you don't have to,
but it's not a bad idea to include this in your `sweep_config` anyway,
in case you change your mind later.
It's also good reproducibility practice to keep note of things like this,
in case you, or someone else,
come back to your Sweep in 6 months or 6 years
and don't know whether `val_G_batch` is supposed to be high or low.
"""


args = Args_Caption(features_dir="data", do_eval=False, output_dir="tmp2")
args.freeze_encoder = False
args.train_tasks = [0,1,0,0]
args.test_tasks = [0,1,0,0]
args.batch_size = 32
args.batch_size_val = 16
args.t1_postprocessing = False
args.datatype = "ourds-CLIP"
parameters_orig_dict = vars(args)

parameters_test_dict = {
    'use_BBX_features':{
        'values': [True, False]
        
    },
    'player_embedding': {
        'values': ["CLIP"]
    },
    'visual_use_diagonal_masking': {
        'values': [True, False]
    },
    'player_embedding_order': {
        'values': ["lineup", "possession"]
    },
    "action_level" : {
        'values': [0]
    }
    }


sweep_config['parameters'] = parameters_test_dict

"""
It's often the case that there are hyperparameters
that we don't want to vary in this Sweep,
but which we still want to set in our `sweep_config`.

In that case, we just set the `value` directly:
"""

parameters_test_dict.update({
    'epochs': {
        'value': 10}
    })

"""
For a `grid` search, that's all you ever need.

For a `random` search,
all the `values` of a parameter are equally likely to be chosen on a given run.

If that just won't do,
you can instead specify a named `distribution`,
plus its parameters, like the mean `mu`
and standard deviation `sigma` of a `normal` distribution.

See more on how to set the distributions of your random variables [here](https://docs.wandb.com/sweeps/configuration#distributions).
"""

"""
When we're finished, `sweep_config` is a nested dictionary
that specifies exactly which `parameters` we're interested in trying
and what `method` we're going to use to try them.
"""

import pprint

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Multimodal-Fusion-Bottleneck")

"""
# Step 3️⃣. Run the Sweep agent
"""



def train(config=None):
    # Initialize a new wandb run
    
    with wandb.init(config=config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        parameters_test_dict = {**parameters_orig_dict, **config}
        run.name="{}_{}_{}_{}_{}".format(parameters_test_dict["task_type"], parameters_test_dict["datatype"], parameters_test_dict["bert_model"], parameters_test_dict["lr"], parameters_test_dict["batch_size"])
        
        main(parameters_test_dict)

"""
The cell below will launch an `agent` that runs `train` 5 times,
usingly the randomly-generated hyperparameter values returned by the Sweep Controller. Execution takes under 5 minutes.
"""

wandb.agent(sweep_id, train, count=30)