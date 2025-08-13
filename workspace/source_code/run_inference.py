from megatron.core.inference.common_inference_params import CommonInferenceParams
import nemo.lightning as nl
from nemo.collections.llm import api
import torch
import argparse

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    sequence_parallel=False,
    setup_optimizers=False,
    
)

trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    ),
)
pr = [
    "### Instruction: Write a summary of the conversation below. ### Input: user: My watchlist is not updating with new episodes (past couple days). Any idea why?\nagent: Apologies for the trouble, Norlene! We're looking into this. In the meantime, try navigating to the season / episode manually.\nuser: Tried logging out/back in, that didn\\u2019t help\nagent: Sorry! \\ud83d\\ude14 We assure you that our team is working hard to investigate, and we hope to have a fix ready soon!\nuser: Thank you! Some shows updated overnight, but others did not...\nagent: We definitely understand, Norlene. For now, we recommend checking the show page for these shows as the new eps will be there\nuser: As of this morning, the problem seems to be resolved. Watchlist updated overnight with all new episodes. Thank you for your attention to this matter! I love Hulu \\ud83d\\udc9a\nagent: Awesome! That's what we love to hear. If you happen to need anything else, we'll be here to support! \\ud83d\\udc9a",
    "### Instruction: Write a summary of the conversation below. ### Input: user: hi , my Acc was linked to an old number. Now I\\u2019m asked to verify my Acc , where a code / call wil be sent to my old number. Any way that I can link my Acc to my current number? Pls help\nagent: Hi there, we are here to help. We will have a specialist contact you about changing your phone number. Thank you.\nuser: Thanks. Hope to get in touch soon\nagent: That is no problem. Please let us know if you have any further questions in the meantime.\nuser: Hi sorry , is it for my account : __email__\nagent: Can you please delete this post as it does have personal info in it. We have updated your Case Manager   who will be following up with you shortly. Feel free to DM us anytime with any other questions or concerns 2/2\nuser: Thank you\nagent: That is no problem. Please do not hesitate to contact us with any further questions. Thank you."
]
prompt = ["### Instruction: Write a summary of the conversation below. ### Input: user: looking to change my flight Friday, Oct 27. GRMSKV to DL4728 from SLC to ORD. Is that an option and what is the cost? Jess\nagent: The difference in fare is $185.30. This would include all airport taxes and fees. The ticket is non-refundable changeable with a fee, *ALS and may result in additional fare collection for changes when making a future changes. *ALS\nuser: I had a first class seat purchased for the original flight, would that be the same with this flight to Chicago?\nagent: Hello, Jess. That is the fare difference. You will have to call us at 1 800 221 1212 to make any changes. It is in First class. *TAY\nuser: thx\nagent: Our pleasure. *ALS\nuser: Do I have to call or is there a means to do this online?\nagent: You can call or you can login to your trip on our website to make changes."]
         

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("query", help="prompting the model")
    #args = parser.parse_args()
    #prompt = args.query
    #print("prompt :", prompt)
    adapter_checkpoint = "/workspace/lab_finetuning_log/checkpoints/model_name=0--val_loss=0.00-step=99-consumed_samples=3200.0-last"  #
    results = api.generate(
    path=adapter_checkpoint,
    prompts=pr,
    trainer=trainer,
    inference_params=CommonInferenceParams(temperature=1, top_k=1, num_tokens_to_generate=100),
    text_only=True,
    )
    nos_of_result= len(results)
    for chat, summary in zip(pr,results):
        top_summary = summary.split("\n")[1]
        print ("Chat History: ", chat, "\n")
        print("=" * 50)
        print("Summary of the Chat ")
        print("=" * 50, '\n')
        print(top_summary)
        print("=" * 50, '\n')
    print(results)
    