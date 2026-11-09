# Med sources

Written by `cite.py` from the card of each source, `SOURCE` in `prepare/<source>.py`: edit the cards, then run `python cite.py`.

The data is not in this repository, and must never be: each user asks for it at its source and accepts its licence. Put each source in its folder under the data root (`MED_DATA`, otherwise `packages/med/data`), then run `python check.py configs/experiment.yaml`: it says what is missing, and whether a file is not the version used here.

| Source | Access | Licence | Folder |
|---|---|---|---|
| [JSRT database, DICOM edition (Japanese Society of Radiological Technology)](http://imgcom.jsrt.or.jp/download/) | open | No licence stated. The page asks to cite the paper below; ask JSRT before sharing the images | `jsrt/` |
| [Montgomery County CXR Set (U.S. National Library of Medicine)](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html) | open | No licence stated. The ReadMe asks not to share the data outside your group, to credit NLM and to cite the papers of the Shenzhen set | `montgomery/CXR_png/` |
| [Nigeria Chest X-ray Dataset, version 1 (Kaggle, 2024)](https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset) | registration | CC BY-NC-SA 4.0, as declared on Kaggle | `nigeria/` |
| [NIH ChestX-ray14, labels of 2020 (Data_Entry_2017_v2020.csv)](https://nihcc.app.box.com/v/ChestXray-NIHCC) | open | No restriction on use (FAQ_CHESTXRAY.pdf): link the NIH download site, acknowledge the NIH Clinical Center and cite the paper | `nih/` |
| [PadChest-GR, grounded reports of 2024-08-19](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) | agreement | PADCHEST research use agreement: research only, no redistribution of any part | `padchest_gr/` |
| [A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh, version 3 (Mendeley Data)](https://data.mendeley.com/datasets/wndbd5r26y/3) | open | CC BY 4.0 | `pneumonia_bangladesh/` |
| [PTX-498, version v2-fix (Zenodo)](https://zenodo.org/records/8266529) | open | CC BY 4.0 | `ptx498/` |
| [RANZCR CLiP - Catheter and Line Position Challenge (Kaggle, 2021), train set](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/data) | registration | Rules of the competition: non-commercial use only, no redistribution | `ranzcr/` |
| [MIDRC-RICORD-1C, version 1 (The Cancer Imaging Archive, 2021)](https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/) | open | CC BY-NC 4.0, and the TCIA data usage policy (cite the data and the paper) | `ricord/` |
| [Shenzhen Hospital CXR Set (U.S. National Library of Medicine)](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html) | open | No licence stated. The ReadMe asks not to share the data outside your group, to credit NLM and Shenzhen No. 3 People's Hospital, and to cite the papers below | `shenzhen/CXR_png/` |
| [Dataset of Tuberculosis Chest X-rays Images, version 3 (Mendeley Data)](https://data.mendeley.com/datasets/8j2g3csprk/3) | open | CC BY 4.0 | `tb_pakistan/` |

Access: **open**, free download; **registration**, an account on the site of the source; **agreement**, a data use agreement to sign; **credentialed**, an identity check and a training before access, as on PhysioNet.

## JSRT database, DICOM edition (Japanese Society of Radiological Technology)

- Script: `prepare/jsrt.py`
- How to get it: Download Nodule154images.zip and NonNodule93images.zip and put them in this folder as they are. Page: http://imgcom.jsrt.or.jp/download/
- Folder checked: `jsrt`, SHA-256 not recorded yet: `python check.py` prints it once the data is there
- Note: The clinical file (CLNDAT) is for registered users only; it is not needed here.

To cite, also in `references.bib`:

```bibtex
@article{shiraishi2000jsrt,
  author = {Shiraishi, Junji and Katsuragawa, Shigehiko and Ikezoe, Junpei and Matsumoto, Tsuneo and Kobayashi, Takeshi and Komatsu, Ken-ichi and Matsui, Mitate and Fujita, Hiroshi and Kodera, Yoshie and Doi, Kunio},
  title = {Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule: Receiver Operating Characteristic Analysis of Radiologists' Detection of Pulmonary Nodules},
  journal = {American Journal of Roentgenology},
  volume = {174},
  number = {1},
  pages = {71--74},
  year = {2000},
  doi = {10.2214/ajr.174.1.1740071}
}
```

## Montgomery County CXR Set (U.S. National Library of Medicine)

- Script: `prepare/montgomery.py`
- How to get it: Open Montgomery-County-CXR-Set/MontgomerySet and download its folder CXR_png (138 PNG) into this folder. Page: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html
- Folder checked: `montgomery/CXR_png`, SHA-256 `b3680373e7102f5446684720c8af72c6228089a251f5723825d19b9d418bc8a8`
- Note: The same three papers as the Shenzhen set: references.bib gives them once.

To cite, also in `references.bib`:

```bibtex
@misc{montgomery2014set,
  author = {{U.S. National Library of Medicine}},
  title = {Montgomery County CXR Set},
  year = {2014},
  howpublished = {\url{https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html}},
  note = {Cite with jaeger2014twopublic, jaeger2014automatic and candemir2014lung}
}
```

## Nigeria Chest X-ray Dataset, version 1 (Kaggle, 2024)

- Script: `prepare/nigeria.py`
- How to get it: Sign in to Kaggle and download the dataset. Put archive.zip in this folder as it is: it is extracted by python -m prepare. The kaggle command names it nigeria-chest-x-ray-dataset.zip: rename it. Page: https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset
- File checked: `nigeria/archive.zip`, SHA-256 not recorded yet: `python check.py` prints it once the data is there
- Note: Its 2,600 file names and sizes are those of two other Kaggle sets, the COVID-19 Radiography Database and the Tuberculosis (TB) Chest X-ray Database: the images may not come from Nigeria (docs/decisions.md).

To cite, also in `references.bib`:

```bibtex
@misc{musa2024nigeria,
  author = {Musa, Aminu and Adamu, Mariya Ibrahim and Kakudi, Habeebah Adamu and Lawal, Yusuf},
  title = {Nigeria Chest X-ray Dataset},
  publisher = {Kaggle},
  year = {2024},
  doi = {10.34740/KAGGLE/DSV/9370352}
}
```

## NIH ChestX-ray14, labels of 2020 (Data_Entry_2017_v2020.csv)

- Script: `prepare/nih.py`
- How to get it: Download Data_Entry_2017_v2020.csv and the 12 archives images/images_001.tar.gz to images_012.tar.gz (batch_download_zips.py does it), then extract the archives in the same folder. Page: https://nihcc.app.box.com/v/ChestXray-NIHCC
- File checked: `nih/Data_Entry_2017_v2020.csv`, SHA-256 not recorded yet: `python check.py` prints it once the data is there
- Note: LOG_CHESTXRAY.pdf lists pixel-identical images, some under two patient numbers.

To cite, also in `references.bib`:

```bibtex
@inproceedings{wang2017chestxray,
  author = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  title = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3462--3471},
  year = {2017},
  doi = {10.1109/CVPR.2017.369}
}
```

## PadChest-GR, grounded reports of 2024-08-19

- Script: `prepare/padchest_gr.py`
- How to get it: Fill in the request form linked from the page and accept the PADCHEST research use agreement. Put grounded_reports_20240819.json, master_table.csv and the PNG images in this folder. Page: https://bimcv.cipf.es/bimcv-projects/padchest-gr/
- File checked: `padchest_gr/grounded_reports_20240819.json`, SHA-256 not recorded yet: `python check.py` prints it once the data is there

To cite, also in `references.bib`:

```bibtex
@article{coelhodecastro2025padchestgr,
  author = {Coelho de Castro, Daniel and Bustos, Aurelia and Bannur, Shruthi and Hyland, Stephanie and Bouzid, Kenza and Wetscherek, Maria Teodora and S{\'a}nchez-Valverde, Maria Dolores and Jaques-P{\'e}rez, Lara and P{\'e}rez-Rodr{\'i}guez, Lourdes and Takeda, Kenji and Salinas-Serrano, Jos{\'e} Mar{\'i}a and Alvarez-Valle, Javier and Galant-Herrero, Joaqu{\'i}n and Pertusa, Antonio},
  title = {PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation},
  journal = {NEJM AI},
  volume = {2},
  number = {7},
  year = {2025},
  doi = {10.1056/AIdbp2401120}
}
```

## A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh, version 3 (Mendeley Data)

- Script: `prepare/pneumonia_bangladesh.py`
- How to get it: Download the archive of version 3 and put it in this folder as it is: it is extracted by python -m prepare. Page: https://data.mendeley.com/datasets/wndbd5r26y/3
- File checked: `pneumonia_bangladesh/Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia.zip`, SHA-256 `3d07a6f0d3ba37c9a72edba45d05e307ab57ebf2ba2effe2b3373da448dcad0c`
- Note: The page announces 3,355 images, the archive holds 2,626. Version 6 (2026) removes duplicates and images whose copies have opposite labels; the paper uses version 3.

To cite, also in `references.bib`:

```bibtex
@misc{hira2026epic,
  author = {Hira, Md Irfanul Kabir and Bithee, Mst Moriom Akter and Ahmed, Shafee and Akter, Laboni and Anonna, Mst. Jannatul Mawa},
  title = {A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh},
  publisher = {Mendeley Data},
  version = {3},
  year = {2026},
  doi = {10.17632/wndbd5r26y.3}
}
```

## PTX-498, version v2-fix (Zenodo)

- Script: `prepare/ptx498.py`
- How to get it: Download PTX-498-v2-fix.zip (MD5 a1574afa9b1f7f299e9fcf142f916b9e on the page) and put it in this folder as it is: it is extracted by python -m prepare. Page: https://zenodo.org/records/8266529
- File checked: `ptx498/PTX-498-v2-fix.zip`, SHA-256 not recorded yet: `python check.py` prints it once the data is there
- Note: Positive only: every image shows a pneumothorax.

To cite, also in `references.bib`:

```bibtex
@article{wang2021deepsdm,
  author = {Wang, Yunpeng and Wang, Kang and Peng, Xueqing and Shi, Lili and Sun, Jing and Zheng, Shibao and Shan, Fei and Shi, Weiya and Liu, Lei},
  title = {DeepSDM: Boundary-aware pneumothorax segmentation in chest X-ray images},
  journal = {Neurocomputing},
  volume = {454},
  pages = {201--211},
  year = {2021},
  doi = {10.1016/j.neucom.2021.05.029}
}
@misc{wang2021ptx498,
  author = {Wang, Yunpeng},
  title = {PTX-498: A multi-center pneumothorax segmentation chest X-ray image dataset},
  publisher = {Zenodo},
  version = {v2-fix},
  year = {2021},
  doi = {10.5281/zenodo.8266529}
}
```

## RANZCR CLiP - Catheter and Line Position Challenge (Kaggle, 2021), train set

- Script: `prepare/ranzcr.py`
- How to get it: Sign in to Kaggle and accept the rules of the competition, then download train.csv and the train/ folder. The test set has no labels. Page: https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/data
- File checked: `ranzcr/train.csv`, SHA-256 not recorded yet: `python check.py` prints it once the data is there
- Note: Its images come from NIH ChestX-ray14, under other names: one image can be in both sources.

To cite, also in `references.bib`:

```bibtex
@article{tang2021clip,
  author = {Tang, Jennifer S. N. and Seah, Jarrel C. Y. and Zia, Adil and others},
  title = {CLiP, catheter and line position dataset},
  journal = {Scientific Data},
  volume = {8},
  pages = {285},
  year = {2021},
  doi = {10.1038/s41597-021-01066-8}
}
@misc{ranzcr2020clip,
  author = {Seah, Jarrel and Jen and Maggie and Law, Meng and Culliton, Phil and Dowd, Sarah},
  title = {RANZCR CLiP - Catheter and Line Position Challenge},
  year = {2020},
  howpublished = {Kaggle, \url{https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification}}
}
```

## MIDRC-RICORD-1C, version 1 (The Cancer Imaging Archive, 2021)

- Script: `prepare/ricord.py`
- How to get it: Download the images with the NBIA Data Retriever and the manifest MIDRC-RICORD-1c-manifest-Jan-2021.tcia, into this folder, then the annotations midrc-ricord-1c-da-other-json.zip, kept as it is. Page: https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/
- File checked: `ricord/midrc-ricord-1c-da-other-json.zip`, SHA-256 not recorded yet: `python check.py` prints it once the data is there

To cite, also in `references.bib`:

```bibtex
@misc{tsai2021ricord1c,
  author = {Tsai, E. and Simpson, S. and Lungren, M. P. and Hershman, M. and Roshkovan, L. and Colak, E. and Erickson, B. J. and Shih, G. and Stein, A. and Kalpathy-Cramer, J. and Shen, J. and Hafez, M. A. F. and John, S. and Rajiah, P. and Pogatchnik, B. P. and Mongan, J. T. and Altinmakas, E. and Ranschaert, E. and Kitamura, F. C. and Topff, L. and Moy, L. and Kanne, J. P. and Wu, C.},
  title = {Data from Medical Imaging Data Resource Center (MIDRC) - RSNA International COVID Radiology Database (RICORD) Release 1c - Chest x-ray, Covid+ (MIDRC-RICORD-1C)},
  publisher = {The Cancer Imaging Archive},
  version = {1},
  year = {2021},
  doi = {10.7937/91ah-v663}
}
@article{tsai2021ricord,
  author = {Tsai, Emily B. and Simpson, Scott and Lungren, Matthew P. and others},
  title = {The RSNA International COVID-19 Open Radiology Database (RICORD)},
  journal = {Radiology},
  volume = {299},
  number = {1},
  pages = {E204--E213},
  year = {2021},
  doi = {10.1148/radiol.2021203957}
}
```

## Shenzhen Hospital CXR Set (U.S. National Library of Medicine)

- Script: `prepare/shenzhen.py`
- How to get it: Open Shenzhen-Hospital-CXR-Set and download its folder CXR_png (662 PNG) into this folder. Page: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html
- Folder checked: `shenzhen/CXR_png`, SHA-256 not recorded yet: `python check.py` prints it once the data is there

To cite, also in `references.bib`:

```bibtex
@article{jaeger2014twopublic,
  author = {Jaeger, Stefan and Candemir, Sema and Antani, Sameer and W{\'a}ng, Yi-Xiang J. and Lu, Pu-Xuan and Thoma, George},
  title = {Two public chest X-ray datasets for computer-aided screening of pulmonary diseases},
  journal = {Quantitative Imaging in Medicine and Surgery},
  volume = {4},
  number = {6},
  pages = {475--477},
  year = {2014},
  doi = {10.3978/j.issn.2223-4292.2014.11.20}
}
@article{jaeger2014automatic,
  author = {Jaeger, Stefan and Karargyris, Alexandros and Candemir, Sema and Folio, Les and Siegelman, Jenifer and Callaghan, Fiona and Xue, Zhiyun and Palaniappan, Kannappan and Singh, Rahul K. and Antani, Sameer and Thoma, George and Wang, Yi-Xiang and Lu, Pu-Xuan and McDonald, Clement J.},
  title = {Automatic Tuberculosis Screening Using Chest Radiographs},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {33},
  number = {2},
  pages = {233--245},
  year = {2014},
  doi = {10.1109/TMI.2013.2284099}
}
@article{candemir2014lung,
  author = {Candemir, Sema and Jaeger, Stefan and Palaniappan, Kannappan and Musco, Jonathan P. and Singh, Rahul K. and Xue, Zhiyun and Karargyris, Alexandros and Antani, Sameer and Thoma, George and McDonald, Clement J.},
  title = {Lung Segmentation in Chest Radiographs Using Anatomical Atlases With Nonrigid Registration},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {33},
  number = {2},
  pages = {577--590},
  year = {2014},
  doi = {10.1109/TMI.2013.2290491}
}
```

## Dataset of Tuberculosis Chest X-rays Images, version 3 (Mendeley Data)

- Script: `prepare/tb_pakistan.py`
- How to get it: Download all the files, then put the two folders Normal Chest X-rays and TB Chest X-rays in this folder. Page: https://data.mendeley.com/datasets/8j2g3csprk/3
- Folder checked: `tb_pakistan`, SHA-256 `a1a8b8d9ab23fbf607b3063827e8a0ca7e8384eb6a84b5b44aeb1776c697bb84`
- Note: 26 pairs of byte-identical files, each within one class.

To cite, also in `references.bib`:

```bibtex
@misc{kiran2026tuberculosis,
  author = {Kiran, Saira and Jabeen, Ishrat},
  title = {Dataset of Tuberculosis Chest X-rays Images},
  publisher = {Mendeley Data},
  version = {3},
  year = {2026},
  doi = {10.17632/8j2g3csprk.3}
}
```
