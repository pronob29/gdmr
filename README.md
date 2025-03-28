# Group-specific Dirichlet Multinomial Regression (gDMR)

## Project Overview

The **Group-specific Dirichlet Multinomial Regression (gDMR)** model is designed to automate the creation of personalized support groups in online health forums by leveraging user-generated content, demographic information, and interaction data. This model extends the **Dirichlet Multinomial Regression (DMR)** framework by incorporating group-specific parameters and **node embeddings** to better capture demographic and interaction-based variations among users.

Through experiments on a large-scale dataset from **MedHelp.com**, we show that **gDMR** outperforms baseline models, including **Latent Dirichlet Allocation (LDA)** and standard DMR, in terms of **topic coherence** and **log-likelihood**. By introducing node embeddings, we further enhance group coherence and semantic quality of generated topics, leading to more meaningful support group formation.

## Key Features

- **Automated Group Formation**: Generates personalized user groups based on content, demographics, and interaction data.
- **Group-Specific Parameters**: Extends DMR to support group-specific parameters for more accurate topic modeling.
- **Node Embeddings**: Enhances the quality and coherence of topics by incorporating interaction-level information through node embeddings.

## Credit and Acknowledgements

This project was developed as part of a PhD research project by **Pronob Kumar Barman**. The repository for this project is maintained on GitHub at **[pronob29/gDMR](https://github.com/pronob29/gdmr)**. This project builds upon the original implementation of Dirichlet Multinomial Regression (DMR) developed by mpkato. We extend their codebase to introduce group-specific modeling capabilities and graph-based user representations.

🔗 Original DMR GitHub Repository: https://github.com/mpkato/dmr

We sincerely thank the original authors for making their implementation publicly available and serving as the foundational basis for our extensions.

## Installation

To install and use the code, clone the repository and install the necessary dependencies:

1. Clone the repository:

    ```bash
    git clone https://github.com/pronob29/gdmr.git
    cd gdmr
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Model Training

To train the **gDMR** model, prepare your text data and covariates, and follow these steps:

```python
from gdmr import gDMR
# Initialize the model with appropriate parameters
gdmr_model = gDMR(G=20, sigma=1.0, beta=0.01, docs=docs_train, covariates=vecs_train, V=vocab_size)

# Train the model
gdmr_model.learning(iteration=1000)
