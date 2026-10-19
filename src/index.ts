/**
 * DSH host plugin registering an image-capable provider backed by Luna visual
 * transcription and one or more existing text-only providers. The whole
 * configuration is also a user settings section (`luna-vision-bridge`), so
 * downstream targets are editable from the Web Settings UI without touching
 * any configuration file.
 */
import type { Context, Volatile } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/cordis-plugin-loader'
import type { SettingsNamespace } from '@deepseek-ai/dsh-settings'
import { LunaVisionBridgeAdapter } from './adapter.js'
import { Config, resolveConfig } from './config.js'
import type { Config as ConfigShape, ResolvedConfig } from './config.js'

export { LunaVisionBridgeAdapter } from './adapter.js'
export type { LunaVisionBridgeDeps } from './adapter.js'
export { Config, resolveConfig } from './config.js'
export type { Config as ConfigShape, ResolvedConfig, ResolvedTarget, TargetConfig } from './config.js'
export { LunaVision, parseCodexJsonl } from './vision.js'
export type { LunaVisionDeps, VisionCommand } from './vision.js'

/** Cordis plugin id used in loader diagnostics. */
export const name = '@ycp424c/dsh-luna-vision-bridge'

/** User-settings namespace surfacing this plugin's configuration in the Web UI. */
export const NS = 'luna-vision-bridge' as SettingsNamespace

/** Services needed to resolve target models and read durable images. */
export const inject = ['llm', 'attachments']

/** Legacy settings hooks; 0.2.1 projects the exported Config through its Loader. */
interface SettingsSectionHooks<T> {
  setSource: (source: () => T) => void
  validate: (value: T) => void
  onChange: () => void
}

/**
 * Register the bridge provider and its settings section. Provider and model
 * selection then appear in the stock DSH model selector without any client
 * plugin, and the Web Settings UI can add downstream targets live.
 * @param ctx - DSH host context carrying LLM and attachment services.
 * @param config - raw loader configuration.
 */
export function apply(ctx: Context, config: ConfigShape | Volatile<ConfigShape> = {}): void {
  const configRef = 'get' in config ? config as Volatile<ConfigShape> : undefined
  let current: () => ConfigShape = () => configRef === undefined ? config as ConfigShape : structuredClone(configRef.get()) as ConfigShape
  // The configuration the adapter serves: a candidate becomes the published
  // value only after every registry refresh succeeds, so a failed route swap
  // keeps serving the last good configuration.
  let published = resolveConfig(current())
  // During a synchronous registry refresh the adapter reads this staged
  // candidate, so `replace` captures the candidate's retry policy and
  // provider metadata instead of the previous generation's.
  let staged: ResolvedConfig | undefined

  const registration = ctx.llm.registerAdapter([published.bridgeProvider], new LunaVisionBridgeAdapter({
    llm: ctx.llm,
    attachments: ctx.attachments,
    config: () => staged ?? published,
  }))
  const directoryEntry = (value: ResolvedConfig) => [{
    provider: value.bridgeProvider,
    displayName: value.providerName,
    settingsNs: NS,
    settingsPath: [],
  }]
  const directory = ctx.llm.registerConfigurableProviders(directoryEntry(published))

  // DSH captures the retry policy at registration, and the policy follows the
  // target provider set (order-insensitive), so a provider-set change must
  // refresh it too.
  const providerFingerprint = (value: ResolvedConfig): string => (
    JSON.stringify([...new Set(value.targets.map(target => target.provider))].sort())
  )

  const sameRegistrationFacts = (a: ResolvedConfig, b: ResolvedConfig): boolean => (
    a.bridgeProvider === b.bridgeProvider
    && a.providerName === b.providerName
    && providerFingerprint(a) === providerFingerprint(b)
  )

  const ensureRegistrationFacts = (): void => {
    let candidate: ResolvedConfig
    try {
      candidate = resolveConfig(current())
    } catch (error) {
      // Beyond-schema constraints failed: refuse the candidate and keep
      // serving the last good configuration, saying so once per snapshot.
      ctx.logger.error('luna-vision-bridge: keeping the last good configuration after an invalid settings section')
      ctx.logger.error(error)
      return
    }
    if (sameRegistrationFacts(candidate, published)) {
      // No registry fact changed (route, provider name, retry-policy source):
      // publish directly; every other field is read live by the adapter.
      published = candidate
      return
    }
    staged = candidate
    try {
      registration.replace([candidate.bridgeProvider])
      try {
        directory.replace(directoryEntry(candidate))
      } catch (error) {
        // Compensate the already-swapped adapter route with the published
        // facts; a refused directory refresh leaves its entries untouched.
        staged = undefined
        try {
          registration.replace([published.bridgeProvider])
        } catch (rollbackError) {
          ctx.logger.error('luna-vision-bridge: route rollback failed after a refused directory refresh; the bridge keeps its previous configuration while the route may have moved')
          ctx.logger.error(rollbackError)
        }
        throw error
      }
    } catch (error) {
      // A refused route swap (e.g. the new bridgeProvider is already owned by
      // another adapter) must not strand the running adapter: the published
      // configuration and the registration keep their previous facts.
      staged = undefined
      ctx.logger.error('luna-vision-bridge: keeping the last good configuration after a failed registration refresh')
      ctx.logger.error(error)
      return
    }
    staged = undefined
    published = candidate
  }

  if (configRef !== undefined) {
    ctx.on('internal/config', function (_raw, next) {
      const raw = next()
      if (this === ctx.fiber) resolveConfig(structuredClone(Config(raw).get()) as ConfigShape)
      return raw
    })
    ctx.on('loader/volatile-update', ensureRegistrationFacts)
  }

  ctx.inject(['settings'], (settingsCtx) => {
    // installSection moved onto the provider in dsh-settings 0.1.2-rc.x
    // (previously a standalone helper taking ctx first). The profile always
    // resolves dsh-settings through the installed dsh, so call the provider
    // method directly.
    type SettingsBridge = Context['settings'] & {
      installSection?: (
        owner: Context,
        ns: SettingsNamespace,
        schema: typeof Config,
        entry: ConfigShape,
        sectionHooks: SettingsSectionHooks<ConfigShape>,
      ) => void
    }
    const provider = settingsCtx.settings as unknown as SettingsBridge
    // New settings forms own the live Config reference; the volatile-update
    // listener above refreshes routes and provider directory without remounting.
    if (typeof provider.installSection !== 'function') return
    provider.installSection(ctx, NS, Config, current(), {
      setSource: (source) => {
        current = source
      },
      validate: (value) => {
        // Reject a section the adapter could not act on: duplicate bridgeModel
        // ids or a target provider colliding with the bridge provider itself.
        resolveConfig(value)
      },
      onChange: ensureRegistrationFacts,
    })
  })
}

// Loader unwraps the default export; keep Config and service dependencies with apply.
export default { name, inject, Config, apply }
