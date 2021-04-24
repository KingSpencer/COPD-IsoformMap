# Improved prediction of smoking status via isoform-awareRNA-seq deep learning models

This repo contains code and necessary files to reproduce the result in the paper, [*Improved prediction of smoking status via isoform-awareRNA-seq deep learning models*](https://www.biorxiv.org/content/10.1101/2020.09.09.290395v1). In this paper, we propose a novel deep learning model, Isoform Map Layer, which maps exon level information to isoform level. Comprehensive experiments on a self collected blood RNA-seq data from 2,557 subjects in the COPDGene Study demonstrate the effectiveness of our method. We hypothesized that since smoking alters40patterns of exon and isoform usage, greater predictive accuracy could be obtained by using exon and isoform-level quantifications to predict smoking status.


## Abstract
Predictive models based on gene expression are already a part of medical decisionmaking for selected situations such as early breast cancer treatment. Most of thesemodels are based on measures that do not capture critical aspects of gene splicing, butwith RNA sequencing it is possible to capture some of these aspects of alternative splicing and use them to improve clinical predictions. Building on previous models topredict cigarette smoking status, we show that measures of alternative splicing significantly improve the accuracy of these predictive models.

## Dependencies
- python 3.6.9
- keras 2.2.4
- numpy 1.16.4
- pandas 0.25.1
- joblib 0.13.2
- tensorflow 1.12.0
- scikit-learn 0.21.2
- (*optinal, for model interpretation*) [DeepExplain](https://github.com/marcoancona/DeepExplain)


## Usage

Please refer to [layer_search.bash](XXX) for the architecture search, and [run.bash](XXX) for running the main experiments. The bash script should be self-explanatory. A sample to run the experiment is shown below: 
```bash
chmod +x run.bash
./run.bash
```

## Trained model weights
We have a folder for saving trained model weights, however, the model weights files are too large to be saved on GitHub. Thus, we provide the following Google Drive link to access the weights we have for our experiments.
- [All model weights](https://drive.google.com/file/d/1XHQXM9cA1IX2jZVp5Hp6LzOCZEihqi9y/view?usp=sharing)

## Code to drive gene, isoform, exon definition
Please see:
- [deeplearning_geneAnnotation.html](https://github.com/KingSpencer/COPD-IsoMap/blob/main/deeplearning_geneAnnotation.html)

## Gene, isoform, exon list and corresponding mapping files
We have save the list of genes, isoforms and exons in this GitHub repo, however, due to the size of the mapping files, we also save them on the external Google Drive.
- [List of genes, isoforms, exons](https://github.com/KingSpencer/COPD-IsoMap/tree/main/mapping_data)
- [Gene-to-exon mapping](https://drive.google.com/file/d/11qG9uAmLuXgL-x3HKR8jRXfUoPXLISkF/view?usp=sharing)
- [Isoform-to-exon mapping](https://drive.google.com/file/d/1jzu9uXVIheKc69kqCp08Kx3AEXdOAWDd/view?usp=sharing)

## Citing the paper
This paper is still under review, please cite the bioRxiv version for now. Once the paper gets published, we will update the citation correspondingly.
```
@article{wang2020improved,
  title={Improved Prediction of Smoking Status via Isoform-Aware RNA-seq Deep Learning Models},
  author={Wang, Zifeng and Masoomi, Aria and Xu, Zhonghui and Boueiz, Adel and Lee, Sool and Zhao, Tingting and Cho, Michael and Silverman, Edwin K and Hersh, Craig and Dy, Jennifer and Castaldi, Peter J},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)