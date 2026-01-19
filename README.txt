I verify that I am the sole author of the programs contained in this archive, except where explicitly stated to the contrary.

Alessio Wu, 15th August 2025

Table of contents
-CFSafety:
    -CFSafety/evaluate_with_gemma3.py
    Contains the python code to evaluate the responses from the models. 

    -CFSafety/evaluate_with_api.py
    Contains the python code to evaluate the subset of responses. 

    -CFSafety/one_shot_reliability_TOST.py
    Tests the validity of one-shot continuous scoring.

    -CFSafety/compare_en_it_box_sig.py
    Plots comparative analysis plots between English and Italian

     -CFSafety/compare_four_ver_box_sig.py
     -CFSafety/compare_three_ver_box_sig.py
     -CFSafety/compare_two_ver_box_sig.py
     Plots the comparative analysis of model versions.

-SFT_XXX (where XXX is the model name and size)
    -SFT_Gemma3_1B/respond_with_XXX.py
    Inference with model base version.

    -SFT_Gemma3_1B/respond_with_PE_XXX.py
    Inference with model prompt engineered version.

    -SFT_Gemma3_1B/train_XXX_qlora_whole_gretel_en.py
    Fine-tune model on English only dataset.

    -SFT_Gemma3_1B/train_XXX_qlora_whole_gretel_en.py
    Fine-tune model on bilingual dataset.

    -SFT_Gemma3_1B/respond_with_safe_XXX_en.py
    Inference with model SFT(Eng) version.

    -SFT_Gemma3_1B/respond_with_safe_XXX_mix.py
    Inference with model SFT(Mix) version.