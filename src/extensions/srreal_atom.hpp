// Mutable Python Atom references without pointers into a resizable vector.
// See LICENSE.rst for license information.
#ifndef SRREAL_ATOM_HPP_INCLUDED
#define SRREAL_ATOM_HPP_INCLUDED

#include <nanobind/nanobind.h>
#include <diffpy/srreal/AtomicStructureAdapter.hpp>
#include "srreal_converters.hpp"

#include <map>
#include <memory>
#include <set>

namespace srrealmodule {

namespace nb = nanobind;
using diffpy::srreal::Atom;
using diffpy::srreal::AtomicStructureAdapter;

class PythonAtom;
struct AtomViewPin;

struct AtomReferences
{
    AtomicStructureAdapter* container;
    std::map<size_t, PythonAtom*> atoms;
    std::set<AtomViewPin*> views;

    explicit AtomReferences(AtomicStructureAdapter* c) : container(c) { }
    ~AtomReferences();
};

inline auto& atom_reference_registry()
{
    static std::map<AtomicStructureAdapter*, std::weak_ptr<AtomReferences>> registry;
    return registry;
}

inline AtomReferences::~AtomReferences()
{
    auto& registry = atom_reference_registry();
    auto it = registry.find(container);
    if (it != registry.end() && it->second.expired()) registry.erase(it);
}

inline std::shared_ptr<AtomReferences> atom_references(AtomicStructureAdapter& c)
{
    auto& entry = atom_reference_registry()[&c];
    auto refs = entry.lock();
    if (!refs) entry = refs = std::make_shared<AtomReferences>(&c);
    return refs;
}

inline std::shared_ptr<AtomReferences> find_atom_references(AtomicStructureAdapter& c)
{
    auto& registry = atom_reference_registry();
    auto it = registry.find(&c);
    return it == registry.end() ? nullptr : it->second.lock();
}

// The Python-visible Atom owns its value, or resolves an index on each access.
// Deletion detaches affected references before changing the container. Other
// references follow index shifts, and growth never invalidates their storage.
class PythonAtom
{
    public:
        PythonAtom() = default;
        explicit PythonAtom(const Atom& a) : mvalue(a) { }
        PythonAtom(const PythonAtom& a) : mvalue(a.value()) { }
        PythonAtom(nb::object owner, size_t index) :
            mowner(std::move(owner)),
            mrefs(atom_references(nb::cast<AtomicStructureAdapter&>(mowner))),
            mindex(index)
        {
            mrefs->atoms.emplace(index, this);
        }

        ~PythonAtom()
        {
            if (mrefs) mrefs->atoms.erase(mindex);
        }

        Atom& value()
        {
            return mrefs ? (*mrefs->container)[static_cast<int>(mindex)] : mvalue;
        }
        const Atom& value() const
        {
            return const_cast<PythonAtom*>(this)->value();
        }

        void detach()
        {
            mvalue = value();
            mrefs->atoms.erase(mindex);
            mrefs.reset();
            mowner.reset();
        }

        void set_index(size_t index) { mindex = index; }
        const std::shared_ptr<AtomReferences>& references() const { return mrefs; }

    private:
        Atom mvalue;
        nb::object mowner;
        std::shared_ptr<AtomReferences> mrefs;
        size_t mindex = 0;
};

inline nb::object atom_reference(nb::object owner, size_t index)
{
    auto refs = atom_references(nb::cast<AtomicStructureAdapter&>(owner));
    auto it = refs->atoms.find(index);
    if (it != refs->atoms.end())
        return nb::cast(it->second, nb::rv_policy::reference);
    return nb::cast(new PythonAtom(std::move(owner), index), nb::rv_policy::take_ownership);
}

// NumPy can retain arbitrary slices of an exported coordinate/Uij view.
// Before a structural edit, preserve the original vector allocation for all
// such views and let the adapter continue with a copied buffer. This avoids
// dangling NumPy data pointers without changing ordinary in-place atom edits.
struct AtomViewPin
{
    std::shared_ptr<AtomReferences> refs;
    std::shared_ptr<AtomicStructureAdapter> snapshot;

    explicit AtomViewPin(std::shared_ptr<AtomReferences> r) : refs(std::move(r))
    {
        refs->views.insert(this);
    }
    ~AtomViewPin()
    {
        if (refs) refs->views.erase(this);
    }
};

inline nb::object pin_atom_view(PythonAtom& atom, nb::object view)
{
    if (atom.references())
    {
        nb::capsule owner(new AtomViewPin(atom.references()), [](void* p) noexcept {
            delete static_cast<AtomViewPin*>(p);
        });
        setNumPyArrayBase(view, std::move(owner));
    }
    return view;
}

inline void preserve_atom_views(const std::shared_ptr<AtomReferences>& refs)
{
    if (refs->views.empty()) return;
    auto snapshot = std::make_shared<AtomicStructureAdapter>(*refs->container);
    // AtomicStructureAdapter has implicit move operations: this transfers the
    // old vector allocation, rather than copying the memory exported to NumPy.
    std::swap(*snapshot, *refs->container);
    for (auto* pin : refs->views)
    {
        pin->snapshot = snapshot;
        pin->refs.reset();
    }
    refs->views.clear();
}

inline void prepare_atom_resize(AtomicStructureAdapter& container)
{
    if (auto refs = find_atom_references(container)) preserve_atom_views(refs);
}

inline void replace_atom_references(AtomicStructureAdapter& container,
        size_t first, size_t last, size_t replacement_size)
{
    auto refs = find_atom_references(container);
    if (!refs) return;
    preserve_atom_views(refs);
    auto atoms = std::move(refs->atoms);
    // Removed or replaced atoms detach. Surviving atoms follow their new
    // indexes, regardless of whether they were obtained by indexing or iteration.
    for (auto [index, atom] : atoms)
    {
        if (first <= index && index < last)
        {
            atom->detach();
            continue;
        }
        size_t shifted = index >= last ? index - (last - first) + replacement_size : index;
        atom->set_index(shifted);
        refs->atoms.emplace(shifted, atom);
    }
}

}  // namespace srrealmodule

#endif
