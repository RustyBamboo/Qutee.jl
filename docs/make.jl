using Documenter
using Qutee

makedocs(
    sitename = "Qutee",
    format = Documenter.HTML(),
    modules = [Qutee]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
