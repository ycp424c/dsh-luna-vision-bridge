import { describe, expect, it, vi } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { updateVolatile } from '@deepseek-ai/cosmokit'
import { Config } from '../src/config.js'
import plugin, { apply } from '../src/index.js'

function harness() {
  const handlers = new Map<string, (...args: any[]) => any>()
  const replace = vi.fn()
  const directoryReplace = vi.fn()
  const ctx = {
    fiber: {},
    llm: {
      registerAdapter: vi.fn(() => ({ replace })),
      registerConfigurableProviders: vi.fn(() => ({ replace: directoryReplace })),
    },
    attachments: {},
    logger: { error: vi.fn() },
    inject: vi.fn(),
    on: vi.fn((event: string, handler: (...args: any[]) => any) => { handlers.set(event, handler) }),
  } as unknown as Context
  return { ctx, handlers, replace, directoryReplace }
}

describe('DSH 0.2.1 live settings', () => {
  it('projects an editable form and refreshes provider registration from the live reference', () => {
    const h = harness()
    const config = Config({})
    expect(Config.meta.volatile).toBe(true)
    expect(plugin.Config).toBe(Config)
    expect(plugin.inject).toEqual(['llm', 'attachments'])
    apply(h.ctx, config)
    updateVolatile(config, Config({ bridgeProvider: 'luna-renamed', providerName: 'Renamed Luna' }))
    h.handlers.get('loader/volatile-update')!()
    expect(h.replace).toHaveBeenCalledWith(['luna-renamed'])
    expect(h.directoryReplace).toHaveBeenCalledWith([expect.objectContaining({
      provider: 'luna-renamed', displayName: 'Renamed Luna', settingsNs: 'luna-vision-bridge',
    })])
  })

  it('rejects invalid live edits before replacing the current reference', () => {
    const h = harness()
    const config = Config({})
    apply(h.ctx, config)
    expect(() => h.handlers.get('internal/config')!.call(h.ctx.fiber, {}, () => ({ providerName: '' }))).toThrow(/providerName/)
    expect(config.get().providerName).toBe('Luna Vision Bridge')
    expect(h.replace).not.toHaveBeenCalled()
  })
})
