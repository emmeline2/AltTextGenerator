## Alternative text generation
to run: python demo.py

dependencies: 
python -m pip install keras, tensorflow, pillow, scipy

Uses the BLIP library, information on that library here:
## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

## Announcement: BLIP is now officially integrated into [LAVIS](https://github.com/salesforce/LAVIS) - a one-stop library for language-and-vision research and applications!

### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre>

### Acknowledgement
The implementation of BLIP relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
