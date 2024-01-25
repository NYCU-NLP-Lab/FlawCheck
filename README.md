# FlawCheck🔍
FlawCheck is a flaw-oriented fact-checking dataset introducted in [How We Refute Claims: Automatic Fact-Checking through Flaw Identification and Explanation]().
Each claim is annotated with maximum four aspects and the explanations of presence or absence of seven flaws: ``Contradicting Facts``, ``Exaggeration``, ``Understatement``, ``Faltering at Times``, ``Lacking Enough Support``, ``Problematic Assumptions``, and ``Existence of Alternative Explanations``.
This dataset encapsulates the expertise of human fact-checking professionals, establishing a new benchmark of flaw-oriented automatic fact-checking. 

## Information⚠️
The paper is currently under the review process.
We now release an example dataset of 500 instances.
The complete dataset will be made available upon the acceptance of this paper.

## Introduction
This dataset is based on a previous work: [WatClaimCheck](https://github.com/nxii/WatClaimCheck).
Given the the demand for premise articles and complete review articles written by human experts, we chose WatClaimCheck as our data source due to its ample and varied collection of premise articles and review articles from eight fact-checking websites. 
We extend WatClaimCheck to study a novel approach of flaw-oriented fact-checking.
We thus collected a total of 33,721 claims in WatClaimCheck to construct FlawCheck.
As the original content in WatClaimCheck included a significant amount of irrelevant web crawl data, we collected the web data again and conducted data cleaning to ensure relatively clean review articles for justification generation evaluation.
We used [GPT-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5) to gather both the aspects and the identified flaws from the review articles.
For more details, please refer to the paper.

In this repo, we provided a direct access to FlawCheck dataset, including generated aspects, flaw explanation, and the renewal review articles.
We use WatClaimCheck's claim index refering to each claim, and metadata like premise articles can also be accessed in WatClaimCheck dataset using the index.
As a result, you have to request the access to WatClaimCheck dataset to use this dataset.

## Usage

### Dataset

**Dataset structure** 

All data is under the ``dataset`` folder, and the file structure looks like this:
```bash
├── dataset
│   ├── train
│   │   ├── aspect
│   │   │   ├── 1.json
│   │   │   ├── 2.json
│   │   │   └── 3.json
│   │   ├── flaw
│   │   └── review
│   ├── dev
|   └── test
```
