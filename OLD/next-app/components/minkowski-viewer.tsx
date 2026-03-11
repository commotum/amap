"use client"

import { useEffect, useMemo, useRef } from "react"
import type p5 from "p5"
import {
  buildMonsterContext,
  clampToGridCoordinate,
  computeHeatmap,
  getGridCoordinateRange,
  getViridisColor,
} from "@/components/math"
import type { MinkowskiControls } from "@/components/types"

interface MinkowskiViewerProps extends MinkowskiControls {
  isOrtho: boolean
  orbitEnabled: boolean
  resetViewKey: number
}

interface HeatmapGeometryScene {
  cellColors: Uint8Array
  gridValue: number
}

interface SketchScene {
  geometryScene: HeatmapGeometryScene
  isOrtho: boolean
  rangeMin: number
  queryX: number
  queryY: number
}

interface OrbitPointerState extends p5 {
  mouseIsPressed: boolean
  movedX: number
  movedY: number
  touches: Array<{ id: number; x: number; y: number }>
}

function readHostSize(host: HTMLDivElement | null) {
  return {
    width: host?.clientWidth ?? 1,
    height: host?.clientHeight ?? 1,
  }
}

function getCellScale(width: number, height: number, gridValue: number) {
  return Math.max(6, Math.floor((Math.min(width, height) * 0.92) / gridValue))
}

function buildCellColors(normalizedScores: Float32Array) {
  const cellColors = new Uint8Array(normalizedScores.length * 3)

  for (let index = 0; index < normalizedScores.length; index += 1) {
    const [red, green, blue] = getViridisColor(normalizedScores[index])
    const colorIndex = index * 3

    cellColors[colorIndex] = red
    cellColors[colorIndex + 1] = green
    cellColors[colorIndex + 2] = blue
  }

  return cellColors
}

function buildHeatmapGeometry(p: p5, scene: HeatmapGeometryScene) {
  const startX = -((scene.gridValue - 1) / 2)
  const startY = -((scene.gridValue - 1) / 2)

  return p.buildGeometry(() => {
    p.noStroke()

    for (let row = 0; row < scene.gridValue; row += 1) {
      for (let col = 0; col < scene.gridValue; col += 1) {
        const scoreIndex = row * scene.gridValue + col
        const colorIndex = scoreIndex * 3

        p.push()
        p.fill(
          scene.cellColors[colorIndex],
          scene.cellColors[colorIndex + 1],
          scene.cellColors[colorIndex + 2]
        )
        p.translate(startX + col, startY + row, 0)
        p.box(1)
        p.pop()
      }
    }
  })
}

function drawSelectedQueryOutline(
  p: p5,
  scene: SketchScene,
  cellScale: number
) {
  const { gridValue } = scene.geometryScene
  const startX = -((gridValue - 1) / 2)
  const startY = -((gridValue - 1) / 2)
  const selectedCol = Math.round(scene.queryX - scene.rangeMin)
  const selectedRow = Math.round(scene.queryY - scene.rangeMin)

  if (
    selectedCol >= 0 &&
    selectedCol < gridValue &&
    selectedRow >= 0 &&
    selectedRow < gridValue
  ) {
    p.push()
    p.noFill()
    p.stroke(241, 245, 249, 220)
    p.strokeWeight(Math.max(2, cellScale * 0.11))
    p.translate(startX + selectedCol, startY + selectedRow, 0)
    p.box(1)
    p.pop()
  }
}

function applyOrbitControl(p: p5, orbitEnabled: boolean) {
  if (orbitEnabled) {
    p.orbitControl(1, 1, 1, { freeRotation: true })
    return
  }

  const pointerState = p as OrbitPointerState
  const previousMouseIsPressed = pointerState.mouseIsPressed
  const previousMovedX = pointerState.movedX
  const previousMovedY = pointerState.movedY
  const previousTouches = pointerState.touches

  pointerState.mouseIsPressed = false
  pointerState.movedX = 0
  pointerState.movedY = 0
  pointerState.touches = []

  p.orbitControl(1, 1, 1, { freeRotation: true })

  pointerState.mouseIsPressed = previousMouseIsPressed
  pointerState.movedX = previousMovedX
  pointerState.movedY = previousMovedY
  pointerState.touches = previousTouches
}

export default function MinkowskiViewer({
  gridValue,
  isOrtho,
  orbitEnabled,
  resetViewKey,
  dimValue,
  thetaValue,
  phiValue,
  extentValue,
  etaValue,
  tValue,
  xValue,
  yValue,
}: MinkowskiViewerProps) {
  const monsterContext = useMemo(
    () =>
      buildMonsterContext({
        gridValue,
        dimValue,
        thetaValue,
        phiValue,
        extentValue,
      }),
    [dimValue, extentValue, gridValue, phiValue, thetaValue]
  )

  const heatmap = useMemo(
    () =>
      computeHeatmap(monsterContext, {
        etaValue,
        tValue,
        xValue,
        yValue,
      }),
    [etaValue, monsterContext, tValue, xValue, yValue]
  )

  const coordinateRange = useMemo(
    () => getGridCoordinateRange(gridValue),
    [gridValue]
  )

  const geometryScene = useMemo<HeatmapGeometryScene>(
    () => ({
      cellColors: buildCellColors(heatmap.normalizedScores),
      gridValue,
    }),
    [gridValue, heatmap.normalizedScores]
  )

  const scene = useMemo<SketchScene>(
    () => ({
      geometryScene,
      isOrtho,
      rangeMin: coordinateRange.min,
      queryX: clampToGridCoordinate(xValue, gridValue),
      queryY: clampToGridCoordinate(yValue, gridValue),
    }),
    [coordinateRange.min, geometryScene, gridValue, isOrtho, xValue, yValue]
  )

  const hostRef = useRef<HTMLDivElement>(null)
  const geometryRef = useRef<p5.Geometry | null>(null)
  const geometrySceneRef = useRef(geometryScene)
  const geometryDirtyRef = useRef(true)
  const orbitEnabledRef = useRef(orbitEnabled)
  const sceneRef = useRef(scene)
  const sketchRef = useRef<p5 | null>(null)

  useEffect(() => {
    sceneRef.current = scene
  }, [scene])

  useEffect(() => {
    geometrySceneRef.current = geometryScene
    geometryDirtyRef.current = true
  }, [geometryScene])

  useEffect(() => {
    orbitEnabledRef.current = orbitEnabled
  }, [orbitEnabled])

  useEffect(() => {
    let resizeObserver: ResizeObserver | null = null
    let disposed = false

    const mountSketch = async () => {
      const p5Module = await import("p5")
      const P5 = p5Module.default
      const host = hostRef.current

      if (disposed || !host) {
        return
      }

      const resizeSketch = (instance: p5) => {
        const { width, height } = readHostSize(hostRef.current)

        if (width <= 0 || height <= 0) {
          return
        }

        instance.resizeCanvas(width, height, true)
      }

      const sketch = (p: p5) => {
        p.setup = () => {
          const { width, height } = readHostSize(host)
          const canvas = p.createCanvas(width, height, p.WEBGL)

          canvas.parent(host)
          canvas.elt.style.display = "block"
          canvas.elt.style.width = "100%"
          canvas.elt.style.height = "100%"

          p.setAttributes("antialias", true)
          p.frameRate(30)
          p.pixelDensity(1)
        }

        p.draw = () => {
          const activeScene = sceneRef.current
          const activeGeometryScene = geometrySceneRef.current
          const cellScale = getCellScale(
            p.width,
            p.height,
            activeGeometryScene.gridValue
          )

          if (geometryDirtyRef.current) {
            if (geometryRef.current) {
              p.freeGeometry(geometryRef.current)
            }

            geometryRef.current = buildHeatmapGeometry(p, activeGeometryScene)
            geometryDirtyRef.current = false
          }

          if (activeScene.isOrtho) {
            p.ortho()
          }

          p.background(0)
          applyOrbitControl(p, orbitEnabledRef.current)
          p.push()
          p.scale(1, -1, 1)
          p.push()
          p.scale(cellScale, cellScale, cellScale)
          p.noStroke()

          if (geometryRef.current) {
            p.model(geometryRef.current)
          }

          drawSelectedQueryOutline(p, activeScene, cellScale)
          p.pop()
          p.pop()
        }

        p.windowResized = () => {
          resizeSketch(p)
        }
      }

      const instance = new P5(sketch, host)

      sketchRef.current = instance

      resizeObserver = new ResizeObserver(() => {
        resizeSketch(instance)
      })
      resizeObserver.observe(host)
    }

    void mountSketch()

    return () => {
      disposed = true
      resizeObserver?.disconnect()

      if (sketchRef.current && geometryRef.current) {
        sketchRef.current.freeGeometry(geometryRef.current)
      }

      geometryRef.current = null
      geometryDirtyRef.current = true
      sketchRef.current?.remove()
      sketchRef.current = null
    }
  }, [isOrtho, resetViewKey])

  return <div ref={hostRef} className="absolute inset-0" />
}
