'use client'

import { useMemo } from 'react'
import p5 from 'p5'
import P5Component from '@/components/p5'
import type { P5Task } from '@/components/tasks/types'
import {
  buildMonsterContext,
  CANVAS_SIZE,
  computeHeatmap,
  getCellSize,
  getViridisColor,
} from '@/components/MonSTERs/Minkowski/math'
import type { MinkowskiControls } from '@/components/MonSTERs/Minkowski/types'

const sketch = (p: p5) => {
  p.setup = () => {
    const canvas = p.createCanvas(canvasDimensions.width, canvasDimensions.height, p.WEBGL)
    canvas.parent(host)
    host.replaceChildren(canvas.elt)
    canvas.elt.style.display = 'block'
    p.smooth()
    if (task.setup) {
      task.setup(p)
    }
  }

  p.draw = () => {
    if (isOrtho) { p.ortho() }
    p.background(0)
    p.orbitControl(1, 1, 1, { freeRotation: true })
    p.scale(1, -1, 1) // Flip Y axis so +Y goes up
    task.draw(p)
  }

interface Minkowski2DtProps extends MinkowskiControls {
  isOrtho: boolean
}

function drawHeatmapGrid(
  p: p5,
  normalizedScores: Float32Array,
  gridValue: number,
  cellSize: number
) {
  const startX = -((gridValue - 1) * cellSize) / 2
  const startY = -((gridValue - 1) * cellSize) / 2

  p.stroke(16, 18, 24)
  p.strokeWeight(1)

  for (let row = 0; row < gridValue; row += 1) {
    for (let col = 0; col < gridValue; col += 1) {
      const index = (row * gridValue) + col
      const [red, green, blue] = getViridisColor(normalizedScores[index])

      p.push()
      p.translate(startX + (col * cellSize), startY + (row * cellSize), 0)
      p.fill(red, green, blue)
      p.box(cellSize, cellSize, cellSize)
      p.pop()
    }
  }
}

export default function Minkowski2Dt({
  gridValue,
  isOrtho,
  dimValue,
  thetaValue,
  phiValue,
  extentValue,
  etaValue,
  tValue,
  xValue,
  yValue,
}: Minkowski2DtProps) {
  const monsterContext = useMemo(
    () =>
      buildMonsterContext({
        gridValue,
        dimValue,
        thetaValue,
        phiValue,
        extentValue,
      }),
    [gridValue, dimValue, thetaValue, phiValue, extentValue]
  )

  const heatmap = useMemo(
    () =>
      computeHeatmap(monsterContext, {
        etaValue,
        tValue,
        xValue,
        yValue,
      }),
    [monsterContext, etaValue, tValue, xValue, yValue]
  )

  const cellSize = useMemo(() => getCellSize(gridValue), [gridValue])

  const task = useMemo<P5Task>(
    () => ({
      width: CANVAS_SIZE,
      height: CANVAS_SIZE,
      setup: (p) => {
        p.noSmooth()
      },
      draw: (p) => {
          drawHeatmapGrid(p, heatmap.normalizedScores, gridValue, cellSize)
      },
    }),
    [cellSize, gridValue, heatmap]
  )

  return <P5Component task={task} isOrtho={isOrtho} />
}
