---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
---

# **polari: how and what to learn**


https://github.com/TomBurdge/polari
https://github.com/TomBurdge/faux_lars
---
---

# Roadmap
1. `DataFrames`: A Genealogy
2. `polars`: What is it?
3. `polari`: Text analytics
![bg right:50% 50%](<https://images.pexels.com/photos/1955134/pexels-photo-1955134.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2>)
---
# 1. DataFrames: A Genealogy
* `Methodology of Scientific Research Programmes` - Imre Lakatosz

* **Science** consists of research programmes.

* ### "A sequence of falsifiable theories characterized by shared a hard core of central theses that are deemed irrefutable—or, at least, refutation-resistant—by methodological fiat." [Imre Lakatos SEP](https://plato.stanford.edu/entries/lakatos/#ImprPoppScie)
---
---
# 1. DataFrames: A Genealogy - `pandas`
* Open-sourced in 2008
* **Extremely** accessible
* Database research has **refuted** pandas:
    * Lazy execution/ Query optimization
    * Join optimization
* And, to some extent, so has use:
    * Sprawling syntax
    * Indexes
---
# 1. DataFrames: A Genealogy - `spark`
* Hadoop and distributed processing
    * "The institutional form in which cooperation is embedded makes all the difference." [Michael Munger](http://www.michaelmunger.com/about.htm)

* scale (`hadoop`) + accessibility (`pandas`) -> `spark`

* `pyspark`
    * "Why don't you just make all the data scientists use pyspark?"
---
# 1. DataFrames: A Genealogy - `duckdb`
* Synthesis: TCPH - reasonable scale.
* "Medium sized data is going to eat everything" - Hannes Mühleisen, co-founder of DuckDB labs
* Thesis: Single-node processing
* Anti-thesis: Parallellism
![bg left:40% 90%](<assets/tcph-example.png>)
---
# 1. DataFrames: `anti-synthesis`
* `Jevons paradox` - efficiency improvements lead to *greater* consumption.
* There will always be large-scale.

* `polars` - GPU & Nvidia.
* `duckdb` - Motherduck.
![bg right:40% 90%](<assets/coal factory.jpg>)
---
# 1. DataFrames: JVM
* `spark-connect`: spark is going multi-lingual
* `sql-frame` (and `ibis`): migrating from spark will become easier
* `photon`: C++
* Persistence to refutation: `Unity Catalogue`
![bg right:40% 90%](<assets/databrics-logo.png>)
---
# 1. DataFrames
* `ibis` & `theseus`
* C++
    * The GPU language
    * DuckDB
* **Rust**:
    * `polars`
    * `DataFusion`
    * `lancedb` & specificity
![bg right:50% 80%](<language screenshot.png>)
---
---
# 2 What is `polars`?

---
# 2 What is `polars`?
* Open-source, single-node DataFrame library.
* Small to medium sized data (up to ~10 million rows).
* Quick.
* Really nice syntax.
---
# 2 What is `polars`?
* Relatively easily extensible.🔌
* https://marcogorelli.github.io/polars-plugins-tutorial/
![bg left:40% 80%](https://marcogorelli.github.io/polars-plugins-tutorial/assets/image.png)

---
# 3 `polari`
---
# 3 `polari`
* https://github.com/TomBurdge/polari
* **Detect the language of natural language text.** 🌐 🔎
* Detect the *sentiment* of English language text. 😀😠
---
# 3 `polari` - trigrams
* "love it"
* `_lov`, `lov`, `ove`, `ve_`, `e_i`, `_it`, `it_`
* More common by language
![bg right:40% 80%](https://images.pexels.com/photos/5650018/pexels-photo-5650018.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)
---
# 3 `polari` - How does it work?
* Three supported algorithms
* Detect the language's script -> Latin/Arabic
* Load into memory the most-common 300 tri-grams by supported language & detected script.
    * Latin -> French/English/Spanish/Esperanto.
* Which set of 300 most-common trigrams are closest to my input?
---
# 3 `polari` An immediate github issue...

![bg centre:40% 80%](assets/github-issue.png)
---
---
# 3 `polari` In Action

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────┬────────────┐
    │ inputs                                                                                                ┆ detected_lang ┆ true_lang  │
    │ ---                                                                                                   ┆ ---           ┆ ---        │
    │ str                                                                                                   ┆ str           ┆ str        │
    ╞═══════════════════════════════════════════════════════════════════════════════════════════════════════╪═══════════════╪════════════╡
    │ Quem pintou o quadro Mona Lisa?                                                                       ┆ Portuguese    ┆ Portuguese │
    │ Takip eden cümleyi tamamlayınız:  Hernández, Meksika millî takımının 2010 FIFA Dünya Kupası ve        ┆ Turkish       ┆ Turkish    │
    │ Apakah maksud "adu untung"?                                                                           ┆ French        ┆ English    │
    │ ¿Qué son las galaxias?                                                                                ┆ Spanish       ┆ Spanish    │
    │ ニューヨーク国民音楽院院長としてアメリカに滞在中に交響曲『新世界より』を作った、チェコの作曲家は誰で  ┆ Japanese      ┆ Japanese   │
    │ しょう？                                                                                              ┆               ┆            │
    │ Medicana SGK geciyor mu?                                                                              ┆ Turkish       ┆ Turkish    │
    │ Hãy tiếp tục đoạn văn sau: Ông chủ của ChatGPT đánh giá đa số dự án tại Thung lũng Silicon sẽ định h… ┆ Vietnamese    ┆ Vietnamese │
    │ Ответьте на вопрос: когда распался Советский Союз?                                                    ┆ Russian       ┆ Russian    │
    │ Một trường trung học cơ sở tổ chức cho lớp 6D gồm 40 học sinh đi tham quan học tập ngoại khóa. Toàn … ┆ Vietnamese    ┆ Vietnamese │
    │ Quais foram as principais contribuições de Bertrand Russell para a filosofia?                         ┆ Portuguese    ┆ Portuguese │
    └───────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────┴────────────┘
```
---

# 3 `polari` In Action
```markdown
┌─────────────────────────────┬───────────────┐
│ inputs                      ┆ detected_lang │
│ ---                         ┆ ---           │
│ str                         ┆ str           │
╞═════════════════════════════╪═══════════════╡
│ Apakah maksud "adu untung"? ┆ Indonesian    │
└─────────────────────────────┴───────────────┘
```
---
# 3 `polari` In Action

# Live Demo!
---
# 3 `polari` In Action
![bg centre:60% 90%](assets/hugging-face-sc.png)
---
---
# 3 `polari`

* Ideas from the corporate sphere...

* Multi-lingual LLMs.
* Model monitoring.