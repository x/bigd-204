import { defineMonacoSetup } from '@slidev/types'

export default defineMonacoSetup(() => {
  return {
    editorOptions: {
      /* Options for monaco-diff */
      renderSideBySide:false,
    }
  }
})