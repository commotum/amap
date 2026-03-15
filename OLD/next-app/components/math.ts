import type { EtaSignature } from "@/components/types"

const MONSTER_SLICE = 12
const QUERY_RANDOM_SEED = 0
const KEY_RANDOM_SEED = 1
const SPATIAL_UNIT = 1
const TEMPORAL_UNIT = 0.005
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
  readonly timeCosh: Float64Array
  readonly timeSinh: Float64Array
  readonly xCos: Float64Array
  readonly xSin: Float64Array
  readonly yCos: Float64Array
  readonly ySin: Float64Array
  readonly constant: number
}

export interface MonsterContext {
  readonly gridValue: number
  readonly dimValue: number
  readonly spatialUnit: number
  readonly temporalUnit: number
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

const EMBEDDING_CACHE = new Map<string, Float64Array>()
const INV_FREQ_CACHE = new Map<string, Float64Array>()
const METRIC_CACHE = new Map<
  number,
  Record<EtaSignature, MonsterMetricCoefficients>
>()

function roundCoordinate(value: number) {
  return Number(value.toFixed(6))
}

function snapToLattice(value: number, min: number, max: number, step: number) {
  const clampedValue = clamp(value, min, max)
  const latticeOffset = (clampedValue - min) / step
  const lower = roundCoordinate(min + Math.floor(latticeOffset) * step)
  const upper = roundCoordinate(min + Math.ceil(latticeOffset) * step)
  const lowerDistance = Math.abs(clampedValue - lower)
  const upperDistance = Math.abs(upper - clampedValue)

  if (lowerDistance < upperDistance) {
    return lower
  }

  if (upperDistance < lowerDistance) {
    return upper
  }

  if (Math.abs(upper) > Math.abs(lower)) {
    return upper
  }

  if (Math.abs(lower) > Math.abs(upper)) {
    return lower
  }

  return clampedValue >= 0 ? upper : lower
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

function getCachedEmbedding(dimValue: number, seed: number) {
  const cacheKey = `${dimValue}:${seed}`
  const cachedEmbedding = EMBEDDING_CACHE.get(cacheKey)

  if (cachedEmbedding) {
    return cachedEmbedding
  }

  const embedding = createRandomEmbedding(dimValue, seed)
  EMBEDDING_CACHE.set(cacheKey, embedding)
  return embedding
}

function getCachedInvFreq(numFreq: number, baseValue: number) {
  const cacheKey = `${numFreq}:${baseValue}`
  const cachedInvFreq = INV_FREQ_CACHE.get(cacheKey)

  if (cachedInvFreq) {
    return cachedInvFreq
  }

  const invFreq = createInvFreq(numFreq, baseValue)
  INV_FREQ_CACHE.set(cacheKey, invFreq)
  return invFreq
}

function buildMetricCoefficients(
  queryBlocks: Float64Array,
  keyBlocks: Float64Array,
  numFreq: number,
  metric: readonly [number, number, number, number]
) {
  const timeCosh = new Float64Array(numFreq)
  const timeSinh = new Float64Array(numFreq)
  const xCos = new Float64Array(numFreq)
  const xSin = new Float64Array(numFreq)
  const yCos = new Float64Array(numFreq)
  const ySin = new Float64Array(numFreq)
  let constant = 0

  const [m0, m1, m2, m3] = metric

  for (let frequencyIndex = 0; frequencyIndex < numFreq; frequencyIndex += 1) {
    const offset = frequencyIndex * MONSTER_SLICE

    const qxt = queryBlocks[offset]
    const qxx = queryBlocks[offset + 1]
    const qxy = queryBlocks[offset + 2]
    const qxz = queryBlocks[offset + 3]

    const qyt = queryBlocks[offset + 4]
    const qyx = queryBlocks[offset + 5]
    const qyy = queryBlocks[offset + 6]
    const qyz = queryBlocks[offset + 7]

    const qzt = queryBlocks[offset + 8]
    const qzx = queryBlocks[offset + 9]
    const qzy = queryBlocks[offset + 10]
    const qzz = queryBlocks[offset + 11]

    const kxt = keyBlocks[offset]
    const kxx = keyBlocks[offset + 1]
    const kxy = keyBlocks[offset + 2]
    const kxz = keyBlocks[offset + 3]

    const kyt = keyBlocks[offset + 4]
    const kyx = keyBlocks[offset + 5]
    const kyy = keyBlocks[offset + 6]
    const kyz = keyBlocks[offset + 7]

    const kzt = keyBlocks[offset + 8]
    const kzx = keyBlocks[offset + 9]
    const kzy = keyBlocks[offset + 10]
    const kzz = keyBlocks[offset + 11]

    timeCosh[frequencyIndex] =
      m0 * qxt * kxt +
      m1 * qxx * kxx +
      m0 * qyt * kyt +
      m2 * qyy * kyy +
      m0 * qzt * kzt +
      m3 * qzz * kzz

    timeSinh[frequencyIndex] =
      -(m0 * qxt * kxx + m1 * qxx * kxt) -
      (m0 * qyt * kyy + m2 * qyy * kyt) -
      (m0 * qzt * kzz + m3 * qzz * kzt)

    xCos[frequencyIndex] = m2 * qxy * kxy + m3 * qxz * kxz
    xSin[frequencyIndex] = -m2 * qxy * kxz + m3 * qxz * kxy

    yCos[frequencyIndex] = m1 * qyx * kyx + m3 * qyz * kyz
    ySin[frequencyIndex] = -m1 * qyx * kyz + m3 * qyz * kyx

    constant += m1 * qzx * kzx + m2 * qzy * kzy
  }

  return {
    timeCosh,
    timeSinh,
    xCos,
    xSin,
    yCos,
    ySin,
    constant,
  }
}

function getCachedMetrics(dimValue: number, numFreq: number) {
  const cachedMetrics = METRIC_CACHE.get(dimValue)

  if (cachedMetrics) {
    return cachedMetrics
  }

  const queryEmbedding = getCachedEmbedding(dimValue, QUERY_RANDOM_SEED)
  const keyEmbedding = getCachedEmbedding(dimValue, KEY_RANDOM_SEED)
  const metrics = {
    "negative-positive": buildMetricCoefficients(
      queryEmbedding,
      keyEmbedding,
      numFreq,
      [-1, 1, 1, 1]
    ),
    "positive-negative": buildMetricCoefficients(
      queryEmbedding,
      keyEmbedding,
      numFreq,
      [1, -1, -1, -1]
    ),
  } satisfies Record<EtaSignature, MonsterMetricCoefficients>

  METRIC_CACHE.set(dimValue, metrics)
  return metrics
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function safeCosh(value: number) {
  const cappedValue = clamp(value, -40, 40)
  return Math.cosh(cappedValue)
}

function safeSinh(value: number) {
  const cappedValue = clamp(value, -40, 40)
  return Math.sinh(cappedValue)
}

export function getGridCoordinateRange(gridValue: number): GridCoordinateRange {
  if (gridValue % 2 === 0) {
    const halfGrid = gridValue / 2

    return {
      min: roundCoordinate(-halfGrid + 0.5),
      max: roundCoordinate(halfGrid - 0.5),
    }
  }

  const min = -Math.floor(gridValue / 2)
  return {
    min,
    max: min + gridValue - 1,
  }
}

export function getGridCoordinateStep() {
  return 1
}

export function clampToGridCoordinate(value: number, gridValue: number) {
  const coordinateRange = getGridCoordinateRange(gridValue)
  const step = getGridCoordinateStep()

  return snapToLattice(value, coordinateRange.min, coordinateRange.max, step)
}

export function buildMonsterContext({
  gridValue,
  dimValue,
  thetaValue,
  phiValue,
  extentValue,
}: BuildMonsterContextInput): MonsterContext {
  if (dimValue % MONSTER_SLICE !== 0) {
    throw new Error(
      `dimValue must be divisible by ${MONSTER_SLICE}. Received ${dimValue}.`
    )
  }

  const numFreq = dimValue / MONSTER_SLICE

  return {
    gridValue,
    dimValue,
    spatialUnit: SPATIAL_UNIT,
    temporalUnit: TEMPORAL_UNIT,
    spatialInvFreq: getCachedInvFreq(numFreq, thetaValue),
    temporalInvFreq: getCachedInvFreq(numFreq, phiValue),
    metrics: getCachedMetrics(dimValue, numFreq),
  }
}

export function computeHeatmap(
  context: MonsterContext,
  { etaValue, tValue, xValue, yValue }: ComputeHeatmapInput
): HeatmapResult {
  const {
    gridValue,
    dimValue,
    spatialUnit,
    temporalUnit,
    spatialInvFreq,
    temporalInvFreq,
  } = context
  const coefficients = context.metrics[etaValue]
  const coordinateRange = getGridCoordinateRange(gridValue)
  const queryX = clampToGridCoordinate(xValue, gridValue)
  const queryY = clampToGridCoordinate(yValue, gridValue)
  const timeDelta = -tValue
  const scoreOffset = coefficients.constant
  const xScores = new Float64Array(gridValue)
  const yScores = new Float64Array(gridValue)
  const normalizedScores = new Float32Array(gridValue * gridValue)
  const scores = new Float32Array(gridValue * gridValue)
  let timeScore = 0

  for (
    let frequencyIndex = 0;
    frequencyIndex < coefficients.timeCosh.length;
    frequencyIndex += 1
  ) {
    const scaledDelta = timeDelta * temporalUnit * temporalInvFreq[frequencyIndex]
    timeScore +=
      coefficients.timeCosh[frequencyIndex] * safeCosh(scaledDelta) +
      coefficients.timeSinh[frequencyIndex] * safeSinh(scaledDelta)
  }

  for (let col = 0; col < gridValue; col += 1) {
    const deltaX = coordinateRange.min + col - queryX
    let xScore = 0

    for (
      let frequencyIndex = 0;
      frequencyIndex < coefficients.xCos.length;
      frequencyIndex += 1
    ) {
      const scaledDelta = deltaX * spatialUnit * spatialInvFreq[frequencyIndex]
      xScore +=
        coefficients.xCos[frequencyIndex] * Math.cos(scaledDelta) +
        coefficients.xSin[frequencyIndex] * Math.sin(scaledDelta)
    }

    xScores[col] = xScore
  }

  for (let row = 0; row < gridValue; row += 1) {
    const deltaY = coordinateRange.min + row - queryY
    let yScore = 0

    for (
      let frequencyIndex = 0;
      frequencyIndex < coefficients.yCos.length;
      frequencyIndex += 1
    ) {
      const scaledDelta = deltaY * spatialUnit * spatialInvFreq[frequencyIndex]
      yScore +=
        coefficients.yCos[frequencyIndex] * Math.cos(scaledDelta) +
        coefficients.ySin[frequencyIndex] * Math.sin(scaledDelta)
    }

    yScores[row] = yScore
  }

  let minScore = Number.POSITIVE_INFINITY
  let maxScore = Number.NEGATIVE_INFINITY
  let minSpatialScore = Number.POSITIVE_INFINITY
  let maxSpatialScore = Number.NEGATIVE_INFINITY
  const normalizer = Math.sqrt(dimValue)
  const timeOffset = timeScore / normalizer

  for (let row = 0; row < gridValue; row += 1) {
    for (let col = 0; col < gridValue; col += 1) {
      const index = row * gridValue + col
      const spatialScore = (xScores[col] + yScores[row] + scoreOffset) / normalizer
      const score = spatialScore + timeOffset

      scores[index] = score
      minSpatialScore = Math.min(minSpatialScore, spatialScore)
      maxSpatialScore = Math.max(maxSpatialScore, spatialScore)
      minScore = Math.min(minScore, score)
      maxScore = Math.max(maxScore, score)
    }
  }

  const spatialRange = maxSpatialScore - minSpatialScore

  if (spatialRange <= Number.EPSILON) {
    normalizedScores.fill(0.5)
  } else {
    for (let index = 0; index < scores.length; index += 1) {
      normalizedScores[index] = (scores[index] - minSpatialScore) / spatialRange
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
    Math.round(lower[0] + (upper[0] - lower[0]) * interpolation),
    Math.round(lower[1] + (upper[1] - lower[1]) * interpolation),
    Math.round(lower[2] + (upper[2] - lower[2]) * interpolation),
  ]
}
