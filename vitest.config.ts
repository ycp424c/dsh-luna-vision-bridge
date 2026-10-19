import { defineConfig } from 'vitest/config'
import { testAliases } from './scripts/dsh-sdk.mjs'

export default defineConfig({
  resolve: { alias: testAliases() },
  test: {
    environment: 'node', include: ['tests/**/*.spec.ts', 'tests/**/*.spec.tsx'],
    server: { deps: { inline: [/dsh-client-store/, /zustand/, /immer/] } },
    restoreMocks: true, clearMocks: true,
  },
})
