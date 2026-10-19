import type { Context } from '@deepseek-ai/cordis'
import type { ClientRemote, ModelProviderGroup, SettingsNamespaceView } from '@deepseek-ai/dsh-api-remotes/client'
import type { SettingsPathOpView } from '@deepseek-ai/dsh-settings/types'
import type { JsonValue } from '@deepseek-ai/dsh-util-values'

export type { ModelProviderGroup, SettingsNamespaceView }

type Result<T> = { ok: true; value: T } | { ok: false; error: { code: string; message: string } }
type Response<T> = Promise<{ result: Result<T> }>

/** The settings editor's wire needs, shared by the old API and the new Remote. */
export interface LunaSettingsApi {
  settings: {
    describe(input: object): Response<{ namespaces: readonly SettingsNamespaceView[]; writable: boolean }>
    update(input: { ns: string; patch: Record<string, JsonValue>; expectedRevision?: number }): Response<SettingsNamespaceView>
    mutate(input: { ns: string; ops: SettingsPathOpView[]; expectedRevision?: number }): Response<SettingsNamespaceView>
  }
  llm: { models(input: object): Response<{ groups: readonly ModelProviderGroup[] }> }
}

/** Keep the editor's concurrency and drafts intact across DSH's wire migration. */
export function createLunaApi(ctx: Pick<Context, 'get'>): LunaSettingsApi {
  const connection = ctx.get('connection') as { api?: LunaSettingsApi }
  if (connection.api !== undefined) return connection.api
  const remote = ctx.get('remote') as ClientRemote
  return {
    settings: {
      describe: async () => ({ result: await remote.settings.describe() }),
      update: async ({ ns, patch, expectedRevision }) => ({ result: await remote.settings.update(ns, patch, expectedRevision) }),
      mutate: async ({ ns, ops, expectedRevision }) => ({ result: await remote.settings.mutate(ns, ops, expectedRevision) }),
    },
    llm: { models: async () => ({ result: await remote.session.modelCatalog() }) },
  }
}
