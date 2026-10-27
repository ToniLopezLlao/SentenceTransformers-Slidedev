---
theme: dracula
title: Sentence Transformers
transition: slide-left
---

# Sentence Transformers

## Pooling

<div class="absolute bottom-10 right-10 text-sm">
  Toni López Llao
</div>

---
theme: dracula

---
# Transformers

<TransformersDiagram/>

---
theme: dracula
---

# Qué és Sentence Transformers?

Una llibreria Python que converteix **frases en vectors numèrics**.

**Casos d'ús:**

- Semantic Search
- Computing Embeddings
- Reranking

**El problema:**

- Keyword search: paraules exactes
- Semantic search: pel significat

::right::

**Exemple:**

Pregunta: "Gat negre"

Resultat:

1. "El gat negre és maco" ✓
2. "El felí negre meulla" ✓
3. "Els cotxes negres" ✗

---
theme: dracula
---

# La pregunta de partida

## Com passa **pooling** de tokens a vector?

<PoolingDiagram/>

Per què? Necessites 1 vector per comparar frases.

---
theme: dracula
---

# Qué és Pooling?

**El problema:**

<pre style="font-family: monospace; background: #1e1e1e; padding: 20px; border-radius: 8px; color: #a8ff60;">
Frase: "Gat negre"
    ↓
BERT tokenitza i processa
    ↓
Token 1: [0.1,  0.2,  0.3]
Token 2: [0.5,  0.6,  0.7]

 Tenim 2 vectors
 Necessitem 1 per comparar frases
</pre>
---
theme: dracula
---

**Estratègies:**

1. **Mean Pooling**: promig
2. **Max Pooling**: màxim
3. **CLS Pooling**: primer token

---
layout: two-cols
---

# Mean Pooling

**Com ho fa:** fa la mitjana de tots els tokens, número a número.

**Per a què serveix:**

- Representació general de tota la frase
- Totes les paraules compten igual
- És l'opció per defecte de Sentence Transformers

<div class="mt-4 text-sm opacity-70">Ideal per a semantic search i similitud entre frases.</div>

::right::

<div class="h-full flex items-center justify-center">
  <PoolingStrategy mode="mean" />
</div>

---
layout: two-cols
---

# CLS Pooling

**Com ho fa:** es queda només amb el vector del token especial `[CLS]`, el primer, i ignora la resta.

**Per a què serveix:**

- Models entrenats perquè el `[CLS]` resumeixi la frase (p. ex. BERT per a classificació)
- Molt ràpid: no cal operar
- Si el model no s'ha entrenat així, el resultat és pitjor

::right::

<div class="h-full flex items-center justify-center">
  <PoolingStrategy mode="cls" />
</div>

---
layout: two-cols
---

# Max Pooling

**Com ho fa:** per a cada posició del vector, es queda amb el valor més alt entre tots els tokens.

**Per a què serveix:**

- Captura les característiques més destacades
- Útil quan una paraula clau marca el significat
- Menys sensible a paraules "de farciment"

::right::

<div class="h-full flex items-center justify-center">
  <PoolingStrategy mode="max" />
</div>

---

# Codi: Classe Pooling

<!-- dentro de las " | | |" pongo las linias que van a ser subralladas i luego la otra estructura ya es etetica de poner le codigo en python y asi evito poner el tag <pre> -->

```python {all|1-2|4-11|13-17|18-23}{lines:true,maxHeight:'420px'}
class Pooling(nn.Module):
    """Pooling layer que fusiona tokens en 1 vector"""

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = "mean",
    ):
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode = pooling_mode

    def forward(self, token_embeddings):
        """
        Entrada: (batch, tokens, 768)
        Sortida: (batch, 768)
        """
        if self.pooling_mode == "mean":
            return token_embeddings.mean(dim=1)
        elif self.pooling_mode == "max":
            return token_embeddings.max(dim=1)[0]
        elif self.pooling_mode == "cls":
            return token_embeddings[:, 0, :]
```

---

# Mètode forward

```python {all|2-3|5-7}{lines:true}
def forward(self, token_embeddings):
    # Input:  (batch, tokens, 768)
    # Output: (batch, 768)

    if self.pooling_mode == "mean":
        return token_embeddings.mean(dim=1)
        # (32, 50, 768) → (32, 768)
```

**Explicació:**

- `dim=1` → opera sobre tokens
- `.mean()` → calcula mitjana
- Resultat: **1 vector per frase**

---

# PyTorch Dimensions

<pre style="font-family: monospace; background: #1e1e1e; padding: 20px; border-radius: 8px; color: #a8ff60;">
# token_embeddings té 3 dimensiones:
# dim=0: batch (32 frases)
# dim=1: tokens (50 paraules per frase)
# dim=2: embedding (768 números per token)

Shape: (32, 50, 768)
        │   │   └─ embedding_dim
        │   └───── num_tokens
        └───────── batch_size

.mean(dim=1) suma sobre els tokens
(32, 50, 768) → (32, 768)
</pre>

---

# Semantic Search

**Pregunta:** "Gats negres"

**Embedding:** [0.12, -0.45, 0.67, ...]

**BD:**

- "El gat negre és maco"
  → [0.11, -0.43, 0.65, ...] ✓ MÉS SIMILAR

- "Els cotxes negres"
  → [0.5, 0.2, -0.1, ...] ✗ MENYS

**Resultat:** Pooling permet comparar vectors

---

# Connexió amb la carrera

| Concepte  | A la carrera  | En pooling            |
| --------- | ------------- | --------------------- |
| Arrays    | Àlgebra       | Vectors de tokens     |
| Reducció  | Estadística   | Molts → 1             |
| Batches   | Deep learning | 32 frases             |
| Derivades | Càlcul        | .mean() diferenciable |
| OOP       | Programació   | Composició            |

---

# Conclusions

✅ Pooling és simple pero potent

✅ Necessari per semantic search

✅ Mean, Max, CLS per casos diferents

✅ Codi obert: git, PR, tests

✅ Connexió amb la carrera

---

# Gràcies!

Preguntes?

---

# Recursos

**Codi:**

- GitHub: sentence-transformers
- Fitxer: pooling.py
- Tests: test_pooling.py

**Docs:**

- deepwiki.com
- pytorch.org
