package cmd

import (
	"fmt"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	fasttext "github.com/bountylabs/go-fasttext"
	"github.com/spf13/cobra"
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
		preds, err := model.Sentencevec(args[0])
		if err != nil {
			fmt.Println(err)
			return
		}
		pp.Println(preds)
	},
}

func init() {
  sentenceCmd.Flags().StringVarP(&modelPath, "model", "m", "", "path to the fasttext model")
	rootCmd.AddCommand(sentenceCmd)
}
