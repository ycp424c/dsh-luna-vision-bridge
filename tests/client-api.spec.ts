import { describe, expect, it, vi } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { createLunaApi } from '../src/client/api.js'

function context(connection: unknown, remote?: unknown): Pick<Context, 'get'> {
  return { get: (name: string) => name === 'connection' ? connection : remote } as Pick<Context, 'get'>
}

describe('settings wire compatibility', () => {
  it('retains the legacy connection API', () => {
    const api = { settings: {}, llm: {} }
    expect(createLunaApi(context({ api }))).toBe(api)
  })

  it('uses the new Remote methods and keeps the revision guard on writes', async () => {
    const view = { ns: 'luna-vision-bridge', revision: 7, value: {} }
    const remote = {
      settings: {
        describe: vi.fn(async () => ({ ok: true, value: { namespaces: [view], writable: true } })),
        update: vi.fn(async () => ({ ok: true, value: view })),
        mutate: vi.fn(async () => ({ ok: true, value: view })),
      },
      session: { modelCatalog: vi.fn(async () => ({ ok: true, value: { groups: [] } })) },
    }
    const api = createLunaApi(context({}, remote))
    expect((await api.settings.describe({})).result).toMatchObject({ ok: true, value: { writable: true } })
    const patch = { providerName: 'My bridge' }
    await api.settings.update({ ns: view.ns, patch, expectedRevision: 7 })
    expect(remote.settings.update).toHaveBeenCalledWith(view.ns, patch, 7)
    const ops = [{ op: 'unset' as const, path: ['targets'] }]
    await api.settings.mutate({ ns: view.ns, ops, expectedRevision: 7 })
    expect(remote.settings.mutate).toHaveBeenCalledWith(view.ns, ops, 7)
    expect((await api.llm.models({})).result).toEqual({ ok: true, value: { groups: [] } })
    expect(remote.session.modelCatalog).toHaveBeenCalledOnce()
  })

  it('preserves a new Remote conflict instead of treating the write as successful', async () => {
    const failure = { ok: false, error: { code: 'settings/conflict', message: 'revision changed' } }
    const api = createLunaApi(context({}, { settings: { update: vi.fn(async () => failure) } }))
    expect((await api.settings.update({ ns: 'luna-vision-bridge', patch: {}, expectedRevision: 2 })).result).toEqual(failure)
  })
})
