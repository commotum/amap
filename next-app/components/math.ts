import type { EtaSignature } from '@/components/MonSTERs/Minkowski/types'

export const CANVAS_SIZE = 512
export const MAX_CELL_SIZE = 32
const MONSTER_SLICE = 12
const RANDOM_SEED = 0
const VIRIDIS_STOPS: ReadonlyArray<readonly [number, number, number]> = [
  [68, 1, 84],
  [71, 44, 122],
  [59, 81, 139],
  [44, 113, 142],
  [33, 144, 141],
  [39, 173, 129],
  [92, 200, 99],
  [170, 220, 50],
  [253, 231, 37],
]

interface MonsterMetricCoefficients {
  readonly time: Float64Array
  readonly x: Float64Array
  readonly y: Float64Array
  readonly constant: number
}

export interface MonsterContext {
  readonly gridValue: number
  readonly dimValue: number
  readonly unit: number
  readonly spatialInvFreq: Float64Array
  readonly temporalInvFreq: Float64Array
  readonly metrics: Record<EtaSignature, MonsterMetricCoefficients>
}

interface BuildMonsterContextInput {
  gridValue: number
  dimValue: number
  thetaValue: number
  phiValue: number
  extentValue: number
}

interface ComputeHeatmapInput {
  etaValue: EtaSignature
  tValue: number
  xValue: number
  yValue: number
}

export interface HeatmapResult {
  readonly scores: Float32Array
  readonly normalizedScores: Float32Array
  readonly minScore: number
  readonly maxScore: number
}

export interface GridCoordinateRange {
  readonly min: number
  readonly max: number
}

function createSeededRandom(seed: number) {
  let state = seed >>> 0

  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function createStandardNormal(random: () => number) {
  let spare: number | null = null

  return () => {
    if (spare !== null) {
      const next = spare
      spare = null
      return next
    }

    let u = 0
    let v = 0

    while (u <= Number.EPSILON) {
      u = random()
    }

    v = random()

    const magnitude = Math.sqrt(-2 * Math.log(u))
    const angle = 2 * Math.PI * v
    spare = magnitude * Math.sin(angle)

    return magnitude * Math.cos(angle)
  }
}

function createRandomEmbedding(dimValue: number, seed: number) {
  const random = createSeededRandom(seed)
  const randomNormal = createStandardNormal(random)
  const embedding = new Float64Array(dimValue)
  let squaredNorm = 0

  for (let index = 0; index < dimValue; index += 1) {
    const value = randomNormal()
    embedding[index] = value
    squaredNorm += value * value
  }

  const scale = Math.sqrt(dimValue / squaredNorm)

  for (let index = 0; index < dimValue; index += 1) {
    embedding[index] *= scale
  }

  return embedding
}

function createInvFreq(numFreq: number, baseValue: number) {
  const invFreq = new Float64Array(numFreq)

  for (let index = 0; index < numFreq; index += 1) {
    invFreq[index] = baseValue ** (-index / numFreq)
  }

  return invFreq
}

function buildMetricCoefficients(
  blocks: Float64Array,
  numFreq: number,
  metric: readonly [number, number, number, number]
) {
  const time = new Float64Array(numFreq)
  const x = new Float64Array(numFreq)
  const y = new Float64Array(numFreq)
  let constant = 0

  const [m0, m1, m2, m3] = metric

  for (let frequencyIndex = 0; frequencyIndex < numFreq; frequencyIndex += 1) {
    const offset = frequencyIndex * MONSTER_SLICE

    const xt = blocks[offset]
    const xx = blocks[offset + 1]
    const xy = blocks[offset + 2]
    const xz = blocks[offset + 3]

    const yt = blocks[offset + 4]
    const yx = blocks[offset + 5]
    const yy = blocks[offset + 6]
    const yz = blocks[offset + 7]

    const zt = blocks[offset + 8]
    const zx = blocks[offset + 9]
    const zy = blocks[offset + 10]
    const zz = blocks[offset + 11]

    time[frequencyIndex] =
      (m0 * xt * xt) +
      (m1 * xx * xx) +
      (m0 * yt * yt) +
      (m2 * yy * yy) +
      (m0 * zt * zt) +
      (m3 * zz * zz)

    x[frequencyIndex] = (m2 * xy * xy) + (m3 * xz * xz)
    y[frequencyIndex] = (m1 * yx * yx) + (m3 * yz * yz)
    constant += (m1 * zx * zx) + (m2 * zy * zy)
  }

  return {
    time,
    x,
    y,
    constant,
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function safeCosh(value: number) {
  const cappedValue = clamp(value, -40, 40)
  return Math.cosh(cappedValue)
}

export function getCellSize(gridValue: number) {
  return Math.max(1, Math.min(MAX_CELL_SIZE, Math.floor(CANVAS_SIZE / gridValue)))
}

export function getGridCoordinateRange(gridValue: number): GridCoordinateRange {
  const min = -Math.floor(gridValue / 2)
  return {
    min,
    max: min + gridValue - 1,
  }
}

export function buildMonsterContext({
  gridValue,
  dimValue,
  thetaValue,
  phiValue,
  extentValue,
}: BuildMonsterContextInput): MonsterContext {
  if (dimValue % MONSTER_SLICE !== 0) {
    throw new Error(`dimValue must be divisible by ${MONSTER_SLICE}. Received ${dimValue}.`)
  }

  const numFreq = dimValue / MONSTER_SLICE
  const embedding = createRandomEmbedding(dimValue, RANDOM_SEED)

  return {
    gridValue,
    dimValue,
    unit: extentValue / gridValue,
    spatialInvFreq: createInvFreq(numFreq, thetaValue),
    temporalInvFreq: createInvFreq(numFreq, phiValue),
    metrics: {
      'negative-positive': buildMetricCoefficients(embedding, numFreq, [-1, 1, 1, 1]),
      'positive-negative': buildMetricCoefficients(embedding, numFreq, [1, -1, -1, -1]),
    },
  }
}

export function computeHeatmap(
  context: MonsterContext,
  { etaValue, tValue, xValue, yValue }: ComputeHeatmapInput
): HeatmapResult {
  const { gridValue, dimValue, unit, spatialInvFreq, temporalInvFreq } = context
  const coefficients = context.metrics[etaValue]
  const coordinateRange = getGridCoordinateRange(gridValue)
  const queryX = clamp(xValue, coordinateRange.min, coordinateRange.max)
  const queryY = clamp(yValue, coordinateRange.min, coordinateRange.max)
  const timeDelta = -tValue
  const scoreOffset = coefficients.constant
  const xScores = new Float64Array(gridValue)
  const yScores = new Float64Array(gridValue)
  const normalizedScores = new Float32Array(gridValue * gridValue)
  const scores = new Float32Array(gridValue * gridValue)
  let timeScore = 0

  for (let frequencyIndex = 0; frequencyIndex < coefficients.time.length; frequencyIndex += 1) {
    timeScore +=
      coefficients.time[frequencyIndex] *
      safeCosh(timeDelta * unit * temporalInvFreq[frequencyIndex])
  }

  for (let col = 0; col < gridValue; col += 1) {
    const deltaX = (coordinateRange.min + col) - queryX
    let xScore = 0

    for (let frequencyIndex = 0; frequencyIndex < coefficients.x.length; frequencyIndex += 1) {
      xScore +=
        coefficients.x[frequencyIndex] *
        Math.cos(deltaX * unit * spatialInvFreq[frequencyIndex])
    }

    xScores[col] = xScore
  }

  for (let row = 0; row < gridValue; row += 1) {
    const deltaY = (coordinateRange.min + row) - queryY
    let yScore = 0

    for (let frequencyIndex = 0; frequencyIndex < coefficients.y.length; frequencyIndex += 1) {
      yScore +=
        coefficients.y[frequencyIndex] *
        Math.cos(deltaY * unit * spatialInvFreq[frequencyIndex])
    }

    yScores[row] = yScore
  }

  let minScore = Number.POSITIVE_INFINITY
  let maxScore = Number.NEGATIVE_INFINITY
  const normalizer = Math.sqrt(dimValue)

  for (let row = 0; row < gridValue; row += 1) {
    for (let col = 0; col < gridValue; col += 1) {
      const index = (row * gridValue) + col
      const score = (timeScore + xScores[col] + yScores[row] + scoreOffset) / normalizer

      scores[index] = score
      minScore = Math.min(minScore, score)
      maxScore = Math.max(maxScore, score)
    }
  }

  const scoreRange = maxScore - minScore

  if (scoreRange <= Number.EPSILON) {
    normalizedScores.fill(0.5)
  } else {
    for (let index = 0; index < scores.length; index += 1) {
      normalizedScores[index] = (scores[index] - minScore) / scoreRange
    }
  }

  return {
    scores,
    normalizedScores,
    minScore,
    maxScore,
  }
}

export function getViridisColor(value: number): [number, number, number] {
  const clampedValue = clamp(value, 0, 1)
  const scaledIndex = clampedValue * (VIRIDIS_STOPS.length - 1)
  const lowerIndex = Math.floor(scaledIndex)
  const upperIndex = Math.min(VIRIDIS_STOPS.length - 1, lowerIndex + 1)
  const interpolation = scaledIndex - lowerIndex
  const lower = VIRIDIS_STOPS[lowerIndex]
  const upper = VIRIDIS_STOPS[upperIndex]

  return [
    Math.round(lower[0] + ((upper[0] - lower[0]) * interpolation)),
    Math.round(lower[1] + ((upper[1] - lower[1]) * interpolation)),
    Math.round(lower[2] + ((upper[2] - lower[2]) * interpolation)),
  ]
}
