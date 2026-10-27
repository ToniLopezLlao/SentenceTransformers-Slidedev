<template>
  <div class="ps" role="img" :aria-label="ariaLabel">
    <!-- Vectors dels tokens -->
    <div class="ps__grid" :style="{ '--dims': dims }">
      <template v-for="(token, r) in tokens" :key="r">
        <div class="ps__label" :class="{ 'is-dim': rowDimmed(r) }">
          {{ token.label }}
        </div>
        <div
          v-for="(value, c) in token.vector"
          :key="c"
          class="ps__cell"
          :class="{ 'is-dim': cellDimmed(r, c), 'is-picked': cellPicked(r, c) }"
          :style="heat(value)"
        >
          {{ fmt(value) }}
        </div>
      </template>
    </div>

    <!-- Operació -->
    <div class="ps__op">
      <div class="ps__arrow" />
      <span class="ps__op-name">{{ info.op }}</span>
    </div>

    <!-- Resultat -->
    <div class="ps__grid" :style="{ '--dims': dims }">
      <div class="ps__label ps__label--out">Frase</div>
      <div
        v-for="(value, c) in result"
        :key="c"
        class="ps__cell ps__cell--out"
        :style="heat(value)"
      >
        {{ fmt(value) }}
      </div>
    </div>

    <div class="ps__formula">{{ formula }}</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  mode: {
    type: String,
    default: 'mean',
    validator: (m) => ['mean', 'max', 'cls'].includes(m),
  },
  tokens: {
    type: Array,
    default: () => [
      { label: '[CLS]', vector: [0.2, 0.9, 0.1, 0.4] },
      { label: 'gat', vector: [0.8, 0.3, 0.5, 0.2] },
      { label: 'negre', vector: [0.5, 0.6, 0.9, 0.0] },
    ],
  },
})

const dims = computed(() => props.tokens[0].vector.length)

const INFO = {
  mean: { op: 'mitjana de cada columna' },
  max: { op: 'màxim de cada columna' },
  cls: { op: 'agafa només el [CLS]' },
}
const info = computed(() => INFO[props.mode])

const column = (c) => props.tokens.map((t) => t.vector[c])

// Fila d'on surt el valor màxim de cada columna
const maxRows = computed(() =>
  Array.from({ length: dims.value }, (_, c) => {
    const col = column(c)
    return col.indexOf(Math.max(...col))
  })
)

const result = computed(() =>
  Array.from({ length: dims.value }, (_, c) => {
    const col = column(c)
    if (props.mode === 'mean') return col.reduce((a, b) => a + b, 0) / col.length
    if (props.mode === 'max') return Math.max(...col)
    return col[0]
  })
)

const rowDimmed = (r) => props.mode === 'cls' && r !== 0
const cellDimmed = (r, c) =>
  rowDimmed(r) || (props.mode === 'max' && maxRows.value[c] !== r)
const cellPicked = (r, c) =>
  (props.mode === 'max' && maxRows.value[c] === r) ||
  (props.mode === 'cls' && r === 0)

const fmt = (v) => v.toFixed(1)

// Com més alt el valor, més intens el color
const heat = (v) => ({ background: `rgba(189, 147, 249, ${0.15 + v * 0.75})` })

// Exemple amb la primera dimensió
const formula = computed(() => {
  const col = column(0).map(fmt)
  const out = fmt(result.value[0])
  if (props.mode === 'mean') return `(${col.join(' + ')}) / ${col.length} = ${out}`
  if (props.mode === 'max') return `max(${col.join(', ')}) = ${out}`
  return `vector del [CLS] = [${props.tokens[0].vector.map(fmt).join(', ')}]`
})

const ariaLabel = computed(
  () =>
    `${props.mode} pooling: ${props.tokens.length} vectors de tokens es ` +
    `redueixen a un sol vector [${result.value.map(fmt).join(', ')}].`
)
</script>

<style scoped>
.ps {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  font-family: ui-monospace, Menlo, monospace;
}

.ps__grid {
  display: grid;
  grid-template-columns: 70px repeat(var(--dims), 52px);
  gap: 5px;
  align-items: center;
}

.ps__label {
  text-align: right;
  padding-right: 6px;
  font-size: 13px;
  color: #f8f8f2;
  transition: opacity 0.3s;
}
.ps__label--out { color: #50fa7b; font-weight: 700; }

.ps__cell {
  height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  border: 2px solid transparent;
  font-size: 13px;
  color: #fff;
  transition: opacity 0.3s, border-color 0.3s;
}
.ps__cell.is-picked { border-color: #ffb86c; }
.ps__cell--out { border-color: #50fa7b; }

.is-dim { opacity: 0.2; }

.ps__op {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-left: 70px;
}
.ps__op-name {
  font-size: 12px;
  color: #ffb86c;
}
.ps__arrow {
  width: 2px;
  height: 28px;
  background: #6272a4;
  position: relative;
}
.ps__arrow::after {
  content: '';
  position: absolute;
  left: 50%;
  bottom: -2px;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top: 8px solid #6272a4;
  border-bottom: 0;
}

.ps__formula {
  margin-top: 8px;
  font-size: 12px;
  color: #8be9fd;
}
</style>
