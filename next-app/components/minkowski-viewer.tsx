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
}

interface SketchScene {
  gridValue: number
  isOrtho: boolean
  normalizedScores: Float32Array
  rangeMin: number
  queryX: number
  queryY: number
}

function readHostSize(host: HTMLDivElement | null) {
  return {
    width: host?.clientWidth ?? 1,
    height: host?.clientHeight ?? 1,
  }
}

function drawHeatmapGrid(p: p5, scene: SketchScene) {
  const cubeSize = Math.max(
    6,
    Math.floor((Math.min(p.width, p.height) * 0.92) / scene.gridValue)
  )
  const spacing = cubeSize
  const startX = -((scene.gridValue - 1) * spacing) / 2
  const startY = -((scene.gridValue - 1) * spacing) / 2

  for (let row = 0; row < scene.gridValue; row += 1) {
    for (let col = 0; col < scene.gridValue; col += 1) {
      const scoreIndex = row * scene.gridValue + col
      const normalizedScore = scene.normalizedScores[scoreIndex]
      const [red, green, blue] = getViridisColor(normalizedScore)

      p.push()
      p.noStroke()
      p.fill(red, green, blue)
      p.translate(startX + col * spacing, startY + row * spacing, 0)
      p.box(cubeSize)
      p.pop()
    }
  }

  const selectedCol = scene.queryX - scene.rangeMin
  const selectedRow = scene.queryY - scene.rangeMin

  if (
    selectedCol >= 0 &&
    selectedCol < scene.gridValue &&
    selectedRow >= 0 &&
    selectedRow < scene.gridValue
  ) {
    p.push()
    p.noFill()
    p.stroke(241, 245, 249, 220)
    p.strokeWeight(Math.max(1.5, cubeSize * 0.08))
    p.translate(
      startX + selectedCol * spacing,
      startY + selectedRow * spacing,
      0
    )
    p.box(cubeSize * 1.1)
    p.pop()
  }
}

export default function MinkowskiViewer({
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
}: MinkowskiViewerProps) {
  const scene = useMemo<SketchScene>(() => {
    const monsterContext = buildMonsterContext({
      gridValue,
      dimValue,
      thetaValue,
      phiValue,
      extentValue,
    })

    const heatmap = computeHeatmap(monsterContext, {
      etaValue,
      tValue,
      xValue,
      yValue,
    })

    const coordinateRange = getGridCoordinateRange(gridValue)

    return {
      gridValue,
      isOrtho,
      normalizedScores: heatmap.normalizedScores,
      rangeMin: coordinateRange.min,
      queryX: clampToGridCoordinate(xValue, gridValue),
      queryY: clampToGridCoordinate(yValue, gridValue),
    }
  }, [
    dimValue,
    etaValue,
    extentValue,
    gridValue,
    isOrtho,
    phiValue,
    tValue,
    thetaValue,
    xValue,
    yValue,
  ])

  const hostRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef(scene)
  const sketchRef = useRef<p5 | null>(null)

  useEffect(() => {
    sceneRef.current = scene
  }, [scene])

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

          if (activeScene.isOrtho) {
            p.ortho()
          }

          p.background(0)
          p.orbitControl(1, 1, 1, { freeRotation: true })
          p.scale(1, -1, 1)
          drawHeatmapGrid(p, activeScene)
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
      sketchRef.current?.remove()
      sketchRef.current = null
    }
  }, [isOrtho])

  return <div ref={hostRef} className="absolute inset-0" />
}
