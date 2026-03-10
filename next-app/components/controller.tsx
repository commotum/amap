import {
  ChevronLeft,
  ChevronRight,
  LocateFixed,
  PanelRightClose,
} from "lucide-react"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { EtaSignature } from "@/components/types"

interface ControllerProps {
  isOrtho: boolean
  onResetView: () => void
  onCollapse: () => void
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
  onResetView,
  onCollapse,
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
    <Card className="relative max-h-[calc(100svh-2rem)] w-[min(22rem,calc(100vw-2rem))] overflow-auto border-white/10 bg-slate-950/72 text-slate-50 shadow-2xl backdrop-blur-xl">
      <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="border-white/10 bg-slate-950/60 hover:bg-slate-900"
          onClick={onResetView}
          aria-label="Reset view"
          title="Reset view"
        >
          <LocateFixed />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          className="border-white/10 bg-slate-950/60 hover:bg-slate-900"
          onClick={onCollapse}
          aria-label="Collapse controller"
          title="Collapse controller"
        >
          <PanelRightClose />
        </Button>
      </div>

      <div className="flex flex-col gap-5 p-4 pt-14">
        <div className="flex items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/5 px-3 py-2">
          <Label
            htmlFor="ortho-view"
            className="text-sm font-medium text-slate-200"
          >
            Orthographic
          </Label>
          <Switch
            id="ortho-view"
            checked={isOrtho}
            onCheckedChange={onOrthoChange}
          />
        </div>

        <div className="space-y-3 rounded-xl border border-white/10 bg-white/5 p-3">
          <p className="text-xs font-medium tracking-[0.18em] text-slate-400 uppercase">
            Presets
          </p>

          <div className="relative flex items-center justify-center">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute left-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Previous grid preset"
              onClick={onGridPrevious}
            >
              <ChevronLeft />
            </Button>
            <div className="rounded-md border border-white/10 px-3 py-1.5 text-sm font-semibold">
              Grid = {gridValue}²
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute right-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Next grid preset"
              onClick={onGridNext}
            >
              <ChevronRight />
            </Button>
          </div>

          <div className="relative flex items-center justify-center">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute left-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Previous dim preset"
              onClick={onDimPrevious}
            >
              <ChevronLeft />
            </Button>
            <div className="rounded-md border border-white/10 px-3 py-1.5 text-sm font-semibold">
              dim = {dimValue.toLocaleString("en-US")}
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute right-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Next dim preset"
              onClick={onDimNext}
            >
              <ChevronRight />
            </Button>
          </div>

          <div className="relative flex items-center justify-center">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute left-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Previous theta preset"
              onClick={onThetaPrevious}
            >
              <ChevronLeft />
            </Button>
            <div className="rounded-md border border-white/10 px-3 py-1.5 text-sm font-semibold">
              theta = {thetaValue.toLocaleString("en-US")}
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute right-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Next theta preset"
              onClick={onThetaNext}
            >
              <ChevronRight />
            </Button>
          </div>

          <div className="relative flex items-center justify-center">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute left-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Previous phi preset"
              onClick={onPhiPrevious}
            >
              <ChevronLeft />
            </Button>
            <div className="rounded-md border border-white/10 px-3 py-1.5 text-sm font-semibold">
              phi = {phiValue.toLocaleString("en-US")}
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute right-0 border-white/10 bg-slate-950/60 hover:bg-slate-900"
              aria-label="Next phi preset"
              onClick={onPhiNext}
            >
              <ChevronRight />
            </Button>
          </div>
        </div>

        <div className="space-y-4 rounded-xl border border-white/10 bg-white/5 p-3">
          <div className="flex items-center gap-4">
            <Label className="w-12 shrink-0 text-slate-300">Extent</Label>
            <Slider
              min={1}
              max={81}
              step={0.1}
              value={[extentValue]}
              onValueChange={([value]) => onExtentChange(value ?? extentValue)}
            />
            <span className="w-10 shrink-0 text-right text-sm font-medium">
              {extentValue.toFixed(1)}
            </span>
          </div>

          <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 px-3 py-2">
            <Label
              htmlFor="t-value"
              className="text-sm font-medium text-slate-300"
            >
              t
            </Label>
            <Input
              id="t-value"
              type="number"
              step={1}
              inputMode="numeric"
              value={tValue}
              onChange={(event) => {
                const nextValue = Number.parseInt(event.target.value, 10)

                if (Number.isNaN(nextValue)) {
                  onTChange(0)
                  return
                }

                onTChange(nextValue)
              }}
              className="h-8 w-24 border-white/10 bg-slate-950/70 text-right shadow-none focus-visible:ring-white/20"
            />
          </div>

          <div>
            <Label className="mb-2 block text-xs font-medium tracking-[0.18em] text-slate-400 uppercase">
              Signature
            </Label>
            <Tabs
              value={etaValue}
              onValueChange={(value) => onEtaChange(value as EtaSignature)}
              className="w-full"
            >
              <TabsList className="w-full border border-white/10 bg-slate-950/70">
                <TabsTrigger
                  value="negative-positive"
                  className="px-2 text-xs leading-none"
                >
                  eta = (-,+,+,+)
                </TabsTrigger>
                <TabsTrigger
                  value="positive-negative"
                  className="px-2 text-xs leading-none"
                >
                  eta = (+,-,-,-)
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>

        <div className="space-y-4 rounded-xl border border-white/10 bg-white/5 p-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium tracking-[0.18em] text-slate-400 uppercase">
              Query
            </p>
            <p className="text-sm text-slate-300">
              x = {xValue}, y = {yValue}
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs tracking-[0.16em] text-slate-400 uppercase">
              <span>x</span>
              <span>
                {queryMin} to {queryMax}
              </span>
            </div>
            <Slider
              min={queryMin}
              max={queryMax}
              step={1}
              value={[xValue]}
              onValueChange={([value]) => onXChange(value ?? xValue)}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs tracking-[0.16em] text-slate-400 uppercase">
              <span>y</span>
              <span>
                {queryMin} to {queryMax}
              </span>
            </div>
            <Slider
              min={queryMin}
              max={queryMax}
              step={1}
              value={[yValue]}
              onValueChange={([value]) => onYChange(value ?? yValue)}
            />
          </div>
        </div>
      </div>
    </Card>
  )
}
