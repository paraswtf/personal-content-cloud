# Search test fixtures

Small photos used by `tests/test_search_engine.py` to check the semantic search model
(SigLIP ViT-B/16) and its relevance threshold, together with the portraits in
`tests/fixtures/`. Each was fetched as the 500 px Wikimedia Commons thumbnail and
re-saved at most 480 px wide (JPEG quality 85). Licence as stated on the Commons file page.

| File | Shows | Commons file | Licence |
|---|---|---|---|
| cat_snow.jpg | a red tabby cat standing in snow | Red_tabby_DLH_cat_in_snow.jpg | CC0 |
| green_tshirt.jpg | a young man in a green T-shirt under a white jacket, at night | Matt_Mullenweg_at_SXSW_2005.jpg | CC0 |
| tractor_men.jpg | two men in light shirts sitting by a green tractor (hard negative for "green shirt") | Marek_Family_-_New_Farmers_for_Generations_(20130620-RD-LSC-0323).jpg | Public domain (USDA photo) |
| dog.jpg | a small white and black dog on a leash | Dog_at_Long_Beach.jpg | CC0 |
| birthday_cake.jpg | a "Happy Birthday Wikipedia" cake with green icing | Birthday_cake_for_Wikipedia.jpg | CC0 |

Commons page for each: `https://commons.wikimedia.org/wiki/File:<Commons file>`.

## Calibration (relevance = sigmoid(117.33 * cos - 12.93), query as "a photo of <query>")

| Query | True matches (relevance) | Highest non-match |
|---|---|---|
| cat | cat_snow 0.043 | dog 0.00012 |
| a cat | cat_snow 0.038 | dog 0.00014 |
| person with green shirt | green_tshirt 0.53 | tractor_men 0.00003 |
| man in a suit with a blue tie | obama_2009 0.056, biden 0.025, obama_2012 0.0025 | harris 0.00005 |
| astronaut | aldrin 0.86, armstrong 0.80, apollo11_crew 0.29 | harris 0.00055 |
| woman | harris 0.0039 | dog 0.00008 |
| dog | dog 0.090 | cat_snow 0.00008 |
| birthday cake | birthday_cake 0.41 | green_tshirt 0.00001 |
| car | (none) | birthday_cake 0.00008 |

The lowest true match is 0.0025 and the highest non-match 0.00055, so the default
threshold (`SEARCH_MIN_RELEVANCE`) is 0.001. Broad words that cover a small part of a
photo ("flag", "beach", "group photo", "man" on the astronaut photos) stay below it.
