import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { EtaSignature } from '@/components/MonSTERs/Minkowski/types'

interface ControllerProps {
  isOrtho: boolean
  onOrthoChange: (checked: boolean) => void
  gridValue: number
  dimValue: number
  thetaValue: number
  phiValue: number
  extentValue: number
  etaValue: EtaSignature
  tValue: number
  xValue: number
  yValue: number
  queryMin: number
  queryMax: number
  onGridPrevious: () => void
  onGridNext: () => void
  onDimPrevious: () => void
  onDimNext: () => void
  onThetaPrevious: () => void
  onThetaNext: () => void
  onPhiPrevious: () => void
  onPhiNext: () => void
  onExtentChange: (value: number) => void
  onEtaChange: (value: EtaSignature) => void
  onTChange: (value: number) => void
  onXChange: (value: number) => void
  onYChange: (value: number) => void
}

export default function Controller({
  isOrtho,
  onOrthoChange,
  gridValue,
  dimValue,
  thetaValue,
  phiValue,
  extentValue,
  etaValue,
  tValue,
  xValue,
  yValue,
  queryMin,
  queryMax,
  onGridPrevious,
  onGridNext,
  onDimPrevious,
  onDimNext,
  onThetaPrevious,
  onThetaNext,
  onPhiPrevious,
  onPhiNext,
  onExtentChange,
  onEtaChange,
  onTChange,
  onXChange,
  onYChange,
}: ControllerProps) {
  return (
    <Card
      className="overflow-hidden bg-transparent"
      style={{
        width: '256px',
        height: '512px',
      }}
    >
      <div className="flex h-full w-full flex-col p-4">
        <div className="flex items-center justify-between gap-4 px-3 py-2">
          <Label htmlFor="ortho-view">Orthographic</Label>
          <Switch
            id="ortho-view"
            checked={isOrtho}
            onCheckedChange={onOrthoChange}
          />
        </div>

        <div className="relative flex items-center justify-center px-3 py-2">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute left-3"
            aria-label="Previous grid preset"
            onClick={onGridPrevious}
          >
            <ChevronLeft />
          </Button>
          <div className="rounded-md border px-3 py-1.5 text-sm font-semibold">
            Grid = {gridValue}²
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute right-3"
            aria-label="Next grid preset"
            onClick={onGridNext}
          >
            <ChevronRight />
          </Button>
        </div>

        <div className="relative flex items-center justify-center px-3 py-2">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute left-3"
            aria-label="Previous dim preset"
            onClick={onDimPrevious}
          >
            <ChevronLeft />
          </Button>
          <div className="rounded-md border px-3 py-1.5 text-sm font-semibold">
            dim = {dimValue.toLocaleString('en-US')}
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute right-3"
            aria-label="Next dim preset"
            onClick={onDimNext}
          >
            <ChevronRight />
          </Button>
        </div>

        <div className="relative flex items-center justify-center px-3 py-2">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute left-3"
            aria-label="Previous theta preset"
            onClick={onThetaPrevious}
          >
            <ChevronLeft />
          </Button>
          <div className="rounded-md border px-3 py-1.5 text-sm font-semibold">
            ϑ = {thetaValue.toLocaleString('en-US')}
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute right-3"
            aria-label="Next theta preset"
            onClick={onThetaNext}
          >
            <ChevronRight />
          </Button>
        </div>

        <div className="relative flex items-center justify-center px-3 py-2">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute left-3"
            aria-label="Previous phi preset"
            onClick={onPhiPrevious}
          >
            <ChevronLeft />
          </Button>
          <div className="rounded-md border px-3 py-1.5 text-sm font-semibold">
            φ = {phiValue.toLocaleString('en-US')}
          </div>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="absolute right-3"
            aria-label="Next phi preset"
            onClick={onPhiNext}
          >
            <ChevronRight />
          </Button>
        </div>

        <div className="flex items-center gap-4 px-3 py-2">
          <Label className="w-12 shrink-0">Extent</Label>
          <Slider
            min={1}
            max={81}
            step={0.1}
            value={[extentValue]}
            onValueChange={([value]) => onExtentChange(value)}
          />
          <span className="w-10 shrink-0 text-right text-sm font-medium">
            {extentValue.toFixed(1)}
          </span>
        </div>

        <div className="flex items-center justify-center px-3 py-2">
          <div className="flex w-full items-center justify-center gap-2 rounded-md border px-3 py-1.5 text-sm font-semibold">
            <span>t =</span>
            <Input
              type="number"
              step={1}
              value={tValue}
              onChange={(event) => {
                const nextValue = Number.parseInt(event.target.value, 10)

                if (Number.isNaN(nextValue)) {
                  onTChange(0)
                  return
                }

                onTChange(nextValue)
              }}
              className="h-auto w-16 border-0 bg-transparent px-0 py-0 text-center text-sm font-semibold shadow-none focus-visible:ring-0"
            />
          </div>
        </div>

        <div className="px-3 py-2">
          <Tabs
            value={etaValue}
            onValueChange={(value) => onEtaChange(value as EtaSignature)}
            className="w-full"
          >
            <TabsList className="grid h-auto w-full grid-cols-2 border bg-transparent p-1">
              <TabsTrigger value="negative-positive" className="px-2 py-1.5 text-xs">
                η = (-,+,+,+)
              </TabsTrigger>
              <TabsTrigger value="positive-negative" className="px-2 py-1.5 text-xs">
                η = (+,-,-,-)
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        <div className="flex flex-col gap-4 px-3 py-4">
          <div className="flex items-center gap-4">
            <span className="w-6 shrink-0 text-sm font-medium">-x</span>
            <Slider
              min={queryMin}
              max={queryMax}
              step={1}
              value={[xValue]}
              onValueChange={([value]) => onXChange(value)}
            />
            <span className="w-6 shrink-0 text-right text-sm font-medium">+x</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="w-6 shrink-0 text-sm font-medium">-y</span>
            <Slider
              min={queryMin}
              max={queryMax}
              step={1}
              value={[yValue]}
              onValueChange={([value]) => onYChange(value)}
            />
            <span className="w-6 shrink-0 text-right text-sm font-medium">+y</span>
          </div>
        </div>
      </div>
    </Card>
  )
}
