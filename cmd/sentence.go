package cmd

import (
	"fmt"

	"github.com/k0kubun/pp"
	"github.com/spf13/cobra"
	"github.com/unknwon/com"

	"github.com/nano-interactive/go-fasttext"
)

// predictCmd represents the predict command
var sentenceCmd = &cobra.Command{
	Use:   "sentence -m [path_to_model] [query]",
	Short: "get a sentence vector",
	Args:  cobra.ExactArgs(1), // make sure that there is only one argument being passed in
	Run: func(cmd *cobra.Command, args []string) {
		if !com.IsFile(modelPath) {
			fmt.Println("the file %s does not exist", modelPath)
			return
		}

		// create a model object
		model := fasttext.Open(modelPath)
		// close the model at the end
		defer model.Close()
		// perform the prediction
		preds := model.Sentencevec(args[0])
		pp.Println(preds)
	},
}

func init() {
	sentenceCmd.Flags().StringVarP(&modelPath, "model", "m", "", "path to the fasttext model")
	rootCmd.AddCommand(sentenceCmd)
}
