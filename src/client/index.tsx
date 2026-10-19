/**
 * Luna Vision Bridge settings section, browser half. Registers one
 * `settings.section` entry under the shell-declared slot so the Web Settings
 * page renders a form for the `luna-vision-bridge` namespace: downstream
 * targets and the bridge display name, saved through the settings domain.
 * The Host owns validation and live application; this face only edits the
 * user layer.
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
// Type-only: pulls the settings shell's SlotMap merge (the 'settings.section' entry).
import type {} from '@deepseek-ai/dsh-client-ui-settings/client'
import { LunaSection } from './LunaSection.tsx'
import type { LunaSectionInjected } from './LunaSection.tsx'
import { createLunaApi } from './api.ts'

/** Required services (cordis fiber inject). The target slot is declared by ui-settings' apply. */
export const inject = ['slots', 'connection', 'remote.settings', 'remote.session']

/**
 * Register the section once the `settings.section` declaration is on the
 * ledger. The injected face only carries the settings wire; the section
 * reads the namespace itself on first render.
 * @param ctx - client root context.
 */
export function apply(ctx: ClientContext): void {
  const api = createLunaApi(ctx)
  const injected = (): LunaSectionInjected => ({ api })

  ctx.slots.inject('settings.section', () => ctx.slots.register({
    name: 'settings.section',
    id: 'luna-vision-bridge',
    order: 20,
    label: 'Luna Vision Bridge',
    inject: injected,
  }, LunaSection))
}
