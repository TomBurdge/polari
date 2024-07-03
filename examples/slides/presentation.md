---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
---

# **polari**

https://github.com/TomBurdge/polari

---

# Roadmap
1. what is `polars`?
2. `polari` - text analytics plugin
3. `polari` in action

---
# 1. What is `polars`?

---
# 1 What is `polars`?
* Open-source, single-node DataFrame library.
* Small to medium sized data.
* Quick.
* Really nice syntax.
---
# 1 What is `polars`?
* Relatively easily extensible.🔌
* https://marcogorelli.github.io/polars-plugins-tutorial/
![bg left:40% 80%](https://marcogorelli.github.io/polars-plugins-tutorial/assets/image.png)

---
# 2 `polari`
---
# 2 `polari`
* https://github.com/TomBurdge/polari
* **Detect the language of natural language text.** 🌐 🔎
* Detect the *sentiment* of English language text. 😀😠
---
# 2 `polari` - trigrams
* "love it"
* `_lov`, `lov`, `ove`, `ve_`, `e_i`, `_it`, `it_`
* More common by language
---
# 2 `polari` - How does it work?
* Three supported algorithms
* Detect the language's script -> Latin/Arabic
* Load into memory the most-common 300 tri-grams by supported language & detected script.
    * Latin -> French/English/Spanish/Esperanto.
* Which set of 300 most-common trigrams are closest to my input?
---
# 3 `polari` In Action

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

# An immediate github issue...
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
